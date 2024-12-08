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
import torch.optim as optim

def get_similarity(similarity, args):
    if similarity == 'L1':
        return torch.nn.L1Loss()
    elif similarity == 'L2':
        return torch.nn.MSELoss()
    elif similarity == 'CKA':
        from cka import CudaCKA
        device = torch.device(args.device)
        cuda_cka = CudaCKA(device)
        loss = cuda_cka.linear_CKA()
        return loss
    else:
        raise ValueError(f'Unknown similarity metric: {similarity}')

def boost(args):
    # Get similarity loss
    loss_similarity = get_similarity(args.similarity, args)
    
    # Load fused model
    fused_model = torch.load(args.model_path_fused)
    
    # Define trainable parameters and optimizer
    trainable_params = [fused_model.alpha]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.6)
    
    # Load fine-tuned model and set to evaluation mode
    finetuned_model = torch.load(args.model_path + args.model_name + '/' + args.task + '/finetuned.pt')
    finetuned_model = finetuned_model.to(args.device)
    finetuned_model.eval()
    
    # Get dataset
    batch_data = get_dataset(args.task, finetuned_model.val_preprocess, location=args.data_path, batch_size=args.batch_size)
    
    # Training loop
    for epoch in range(args.steps):
        for i, data in enumerate(tqdm.tqdm(batch_data)):
            start_time = time.time()
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            
            # Forward pass
            outputs, features, _, _ = fused_model(x, args.task)
            finetuned_features = fused_model(x).detach()
            
            # Compute loss
            loss = loss_similarity(features, finetuned_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Log every 20 batches
            if i % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                image_encoder = finetuned_model.get_image_encoder()
                classification_head = finetuned_model.get_classification_head(args.task)
                metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, args.task, args)
                
                log.info(f'Epoch: {epoch}, Iteration: {i}, Dataset: {args.task}, ACC: {metrics["top1"]}, Loss: {loss.item()}, LR: {current_lr}, Time: {elapsed_time:.2f}s')
    
    return fused_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Merge Task Specific Models')
    parser.add_argument('--model', type=str, default='ViT-B-32', help='Model name')
    parser.add_argument('--model_path', type=str, default='models/', help='Path to the model directory')                       # default='models/'
    parser.add_argument('--data_path', type=str, default='datasets/', help='Path to the dataset directory')     # default='datasets/'
    parser.add_argument('--task', type=str, default='SUN397', help='task to be boosted')
    parser.add_argument('--model_path_fused', type=str, default='fused_model_path/', help='Path to the fused model directory')    
    parser.add_argument('--steps', type=int, default='200', help='Number of steps for boosting')       
    parser.add_argument('--similarity', type=str, default='L1', help='Similarity metric for model specialization. (L1, L2, CKA)')
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    log_path = 'logs/' + args.task

    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(log_path, 'log_{}_{}_{}_{}.txt'.format(args.granularity, args.fuse_backbone, args.model, str_time_))
    fused_model_boosted = boost(args)
    torch.save(fused_model_boosted, args.save + '/fused_model_boosted_{}_{}_{}_{}.pt'.format(args.steps, args.lr, args.similarity, str_time_))

    


    