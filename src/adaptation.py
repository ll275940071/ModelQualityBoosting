import time
import tqdm
from filter import *
from model import *
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle
from eval import eval_single_dataset, eval_single_dataset_head, eval_single_dataset_preprocess_head
from modeling import ImageClassifier
import sys
import numpy as np
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
def get_source_model(args, dataset_name):
    finetuned_checkpoint = args.model_path + args.model+'/'+ dataset_name +'/finetuned.pt'
    finetuned_model = torch.load(finetuned_checkpoint)
    image_encoder = finetuned_model
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)
    model = model.to(args.device)
    model.eval()
    return model

def softmax_entropy(fused):
    p = fused.softmax(dim=1)
    log_p = fused.log_softmax(dim=1)
    return -(p * log_p).sum(dim=1)

def JSM(source, fused):
    def cross_loss(source, fused):
        p_source = source.softmax(dim=1)
        log_p_fused = fused.log_softmax(dim=1)
        return -(p_source * log_p_fused).sum(dim=1)
    return cross_loss(source, fused) + softmax_entropy(fused)

def adapt_JSM_filter(fused_model: FusedModel, pre_model, args, log):

    print('-----The initial fusion indices-----')
    print(fused_model.lambdas())
    print('------------------------------------')
    print('-----Trainable_params-----')
    print(list(fused_model.collect_trainable_params()))

    optimizer = torch.optim.Adam(fused_model.collect_trainable_params(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
    for epoch in range(args.epoch):
        losses = 0.
        for dataset_name in args.datasets:
            source_model = get_source_model(args, dataset_name)
            dataset = get_dataset(dataset_name, pre_model.val_preprocess, location=args.data_path, batch_size=args.batch_size)
            dataloader = get_dataloader_shuffle(dataset)
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                data = maybe_dictionarize(data)
                x = data['images'].to(args.device)
                y = data['labels'].to(args.device)
                # with torch.no_grad():
                fused_outputs = fused_model(x, dataset_name)
                source_outputs = source_model(x)
                if args.filter_type=='threshold' or args.filter_type=='percentile':
                    mask, coff = sample_filter(fused_outputs, dataset_name, args)
                    loss = JSM(source_outputs[~mask], fused_outputs[~mask]).mean(0)
                    if mask.sum().item() > 0:
                        weight_factor = args.batch_size / (args.batch_size - mask.sum().item())   # weight factor to compensate for the filtering
                    else:
                        weight_factor = 1.0
                    loss = loss * weight_factor * coff
                else:
                    loss = JSM(source_outputs, fused_outputs).mean(0)
                losses += loss 
                if i > 0: # Execute only one step
                    break
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print('Epoch:', epoch, ': fusion indices:', fused_model.lambdas_raw)
        if ((epoch+1) % 100) == 0:
            log.info(str(list(fused_model.lambdas().data)))
            str_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            file_prefix = f'results/lambdas_{args.granularity}_{args.fuse_backbone}_{args.filter_type}_{str_time}_epoch_{epoch + 1}'
            save_lambdas(fused_model, f'{file_prefix}.pth', f'{file_prefix}.txt')
            Total_ACC = 0.
            for dataset_name in args.datasets:
                image_encoder = fused_model.get_image_encoder()
                classification_head = fused_model.get_classification_head(dataset_name)
                metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
                Total_ACC += metrics['top1']
                log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
            log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(args.datasets)) + '\n')

    return fused_model  

def adapt_JSM(fused_model: FusedModel, pre_model, args, log):

    print('-----The initial fusion indices-----')
    print(fused_model.lambdas())
    print('------------------------------------')
    print('-----Trainable_params-----')
    print(list(fused_model.collect_trainable_params()))

    optimizer = torch.optim.Adam(fused_model.collect_trainable_params(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
    for epoch in range(args.epoch):
        losses = 0.
        for dataset_name in args.datasets:
            source_model = get_source_model(args, dataset_name)
            dataset = get_dataset(dataset_name, pre_model.val_preprocess, location=args.data_path, batch_size=args.batch_size)
            dataloader = get_dataloader_shuffle(dataset)
            for i, data in enumerate(tqdm.tqdm(dataloader)):
                data = maybe_dictionarize(data)
                x = data['images'].to(args.device)
                y = data['labels'].to(args.device)
                # with torch.no_grad():
                merged_outputs = fused_model(x, dataset_name)
                source_outputs = source_model(x)
                loss = JSM(source_outputs, merged_outputs).mean(0)
                losses += loss
                if i > 0: # Execute only one step
                    break
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print('Epoch:', epoch, ': fusion indices:', fused_model.lambdas_raw)

        if ((epoch+1) % 100) == 0:
            log.info(str(list(fused_model.lambdas().data)))
            str_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            file_prefix = f'results/lambdas_{args.granularity}_{args.fuse_backbone}_{args.filter_type}_{str_time}_epoch_{epoch + 1}'
            save_lambdas(fused_model, f'{file_prefix}.pth', f'{file_prefix}.txt')
            Total_ACC = 0.
            for dataset_name in args.datasets:
                image_encoder = fused_model.get_image_encoder()
                classification_head = fused_model.get_classification_head(dataset_name)
                metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
                Total_ACC += metrics['top1']
                log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
            log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(args.datasets)) + '\n')

    return fused_model  


def adapt_EM(fused_model: FusedModel, pre_model, args, log):

    print('-----The initial fusion indices-----')
    print(fused_model.lambdas())
    print('------------------------------------')
    print('-----Trainable_params-----')
    print(list(fused_model.collect_trainable_params()))

    optimizer = torch.optim.Adam(fused_model.collect_trainable_params(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.)
    for epoch in range(args.epoch):
        losses = 0.
        for dataset_name in args.datasets:
            with SuppressOutput():
                dataset = get_dataset(dataset_name, pre_model.val_preprocess, location=args.data_path, batch_size=args.batch_size)
                dataloader = get_dataloader_shuffle(dataset)
            # for i, data in enumerate(tqdm.tqdm(dataloader)):
            for i, data in enumerate(dataloader):
                data = maybe_dictionarize(data)
                x = data['images'].to(args.device)
                y = data['labels'].to(args.device)
                outputs = fused_model(x, dataset_name)
                loss = EM(outputs).mean(0)
                losses += loss
                if i > 0: # Execute only one step
                    break
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        # print('Epoch:', epoch, ': fusion indices:', fused_model.lambdas())
        print('Epoch:', epoch, ': fusion indices:', fused_model.lambdas_raw)

        if (epoch+1 % 100) == 0:
            log.info(str(list(fused_model.lambdas().data)))
            str_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
            file_prefix = f'results/lambdas_{args.granularity}_{args.fuse_backbone}_{args.filter_type}_{str_time}_epoch_{epoch + 1}'
            save_lambdas(fused_model, f'{file_prefix}.pth', f'{file_prefix}.txt')
            Total_ACC = 0.
            for dataset_name in args.datasets:
                image_encoder = fused_model.get_image_encoder()
                classification_head = fused_model.get_classification_head(dataset_name)
                metrics = eval_single_dataset_preprocess_head(image_encoder, classification_head, dataset_name, args)
                Total_ACC += metrics['top1']
                log.info('Eval: Epoch: ' + str(epoch) + ' dataset: ' + str(dataset_name) + ' ACC: ' + str(metrics['top1']))
            log.info('Eval: Epoch: ' + str(epoch) + ' Avg ACC:' + str(Total_ACC / len(args.datasets)) + '\n')

    return fused_model   

def save_lambdas(fused_model, file_path_pth, file_path_txt):
    os.makedirs(os.path.dirname(file_path_pth), exist_ok=True)
    os.makedirs(os.path.dirname(file_path_txt), exist_ok=True)

    lambdas = fused_model.lambdas()
    torch.save(lambdas, file_path_pth)
    print(f"Lambdas have been saved to {file_path_pth}")
    lambdas_np = lambdas.cpu().detach().numpy()   # Convert to numpy array
    np.savetxt(file_path_txt, lambdas_np)
    print(f"Lambdas have been saved to {file_path_txt}")