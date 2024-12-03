import argparse
import time
import torch
import tqdm
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('/home/xuyuebin/Documents/01/submission/ModeBoost')
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# src_dir = os.path.join(parent_dir, 'src')
# sys.path.append(src_dir)
from adaptation import *
from aux import *
from model import *
from task_vector import TaskVector
from datasets.registry import get_dataset
from ties_merging_utils import *
from dare_utils import *  
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle
from eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head
from typing import List


def prune(task_vectors: List[TaskVector], args):
    return 0


def fuse(args):

    # if args.fuse_backbone == 'TA':
    pretrained_path = args.model_path + args.model + '/zeroshot.pt'
    pretrained_model = torch.load(pretrained_path)
    pretrained_model_dict = pretrained_model.state_dict()  # ptm_check

    finetuned_dict = [] 
    task_vectors = [TaskVector(pretrained_checkpoint=pretrained_path, 
                               finetuned_checkpoint=args.model_path + args.model + '/' + dataset_name + '/finetuned.pt',
                               finetuned_dict=finetuned_dict) 
                               for dataset_name in args.datasets]

    paramslist = []
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dict.items())]                            


    if args.dare == True:
        print('DARE start: ', 'mask_rate:', args.mask_rate, 'rescale_flag:', args.rescale_flag, 'mask_strategy:', args.mask_strategy, '\n')
        task_vectors = dare(task_vectors, mask_rate=args.mask_rate, rescale_flag=args.rescale_flag, mask_strategy=args.mask_strategy)
        print('*****DARE finished*****\n')

    if args.fuse_backbone == 'TA':
        paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors

    if args.fuse_backbone == 'Ties':
        remove_keys = []
        flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in finetuned_dict])
        flat_ptm = state_dict_to_vector(pretrained_model_dict, remove_keys)
        tv_flat_checks = flat_ft - flat_ptm
        assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, pretrained_model_dict, remove_keys), pretrained_model_dict)
        assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], pretrained_model_dict, remove_keys), finetuned_dict[i])for i in range(len(finetuned_dict))])
        K = 20
        merge_func = "dis-sum"
        selected_entries, merged_tv = ties_merging_split(tv_flat_checks, reset_thresh=K, merge_func=merge_func,)
        ties_task_vectors = []
        for vector_ in selected_entries:
            t_state_dict = vector_to_state_dict(vector_, pretrained_model_dict, remove_keys=remove_keys)
            ref_model = torch.load(pretrained_path)
            ref_model.load_state_dict(t_state_dict, strict=False)
            ties_task_vectors.append(ref_model.state_dict())
        paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.items())  for i, tv in enumerate(ties_task_vectors)] # task vectors

    ptm_model = CustomModelWrapper(pretrained_model, args.datasets).to(args.device)
    orig_params, names = functionalize(ptm_model)

    torch.cuda.empty_cache()
    fused_model = FusedModel(args, ptm_model, paramslist, names)
    
    # if args.granularity == 'task':
    #     model = ModelWrapper(pretrained_model)
    # torch.cuda.empty_cache()
    return fused_model, pretrained_model


def boost(args):
    return 0    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Merge Task Specific Models')

    parser.add_argument('--model', type=str, default='ViT-B-32', help='Model name')
    parser.add_argument('--model_path', type=str, default='/home/xuyuebin/Documents/models/', help='Path to the model directory')                       # default='models/'
    parser.add_argument('--datasets', type=str, default='SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD', help='Comma-separated list of datasets')   
    parser.add_argument('--data_path', type=str, default='/home/xuyuebin/Documents/datasets/merging_dataset', help='Path to the dataset directory')     # default='datasets/'
    parser.add_argument('--granularity', type=str, default='layer', help='Granularity level (task or layer)')
    parser.add_argument('--fuse_backbone', type=str, default='TA', help='Fusion backbone method (TA, Ties)')
    parser.add_argument('--dare', action='store_true', help='Enable DARE mode')
    parser.add_argument('--mask_rate', type=float, default=0.5, help='Mask rate for DARE mode')
    parser.add_argument('--rescale_flag', action='store_true', help='Enable rescale flag for DARE mode')
    parser.add_argument('--mask_strategy', type=str, default='random', help='Mask strategy for DARE mode')
    parser.add_argument('--filter_type', type=str, default='None', help='Filter type (threshold, percentile, None)')
    parser.add_argument('--EM', action='store_true', help='Enable entropy minimization')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--openclip_cachedir', type=str, default='./.cache/open_clip', help='OpenCLIP cache Directory')

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.datasets = args.datasets.split(',')
    args.save = args.model_path + args.model
    args.openclip_cachedir = args.save
    log_path = 'logs/' + args.model

    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(log_path, 'log_{}_{}_{}_{}.txt'.format(args.granularity, args.fuse_backbone, args.model, str_time_))

    fused_model, pretrained_model = fuse(args)

    fused_model_adapted = adapt_EM(fused_model, pretrained_model, args, log)
    # fused_model_adapted = adapt_JSM(fused_model, pretrained_model, args, log)
    # fused_model_adapted = adapt_JSM_filter(fused_model, pretrained_model, args, log)
    torch.save(fused_model_adapted, args.save + '/fused_model_{}_{}_{}_{}.pt'.format(args.granularity, args.fuse_backbone, args.filter_type, str_time_))
    Total_ACC = 0.
    for dataset_name in args.datasets:
            image_encoder = fused_model.get_image_encoder()
            classification_head = fused_model.get_classification_head(dataset_name)
            metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
            Total_ACC += metrics['top1']
            log.info('Eval: Epoch: ' + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
    log.info('Eval: Epoch: ' + ' Avg ACC:' + str(Total_ACC / len(args.datasets)) + '\n')
    


    