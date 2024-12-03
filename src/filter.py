import math
# from adaptation import *
import torch
datasets_classes = {
    'SUN397': 397,  
    'Cars': 196,
    'RESISC45': 45,
    'EuroSAT': 10,
    'SVHN': 10,
    'GTSRB': 43,
    'MNIST': 10,
    'DTD': 47
}
ent_margin_e0 = 0.9  # 0.4
ent_margin_e0 = {key: ent_margin_e0 * math.log(value) for key, value in datasets_classes.items()}
def softmax_entropy(fused):
    p = fused.softmax(dim=1)
    log_p = fused.log_softmax(dim=1)
    return -(p * log_p).sum(dim=1)

def sample_filter(fused_outputs, dataset_name, args):
    entropy = softmax_entropy(fused_outputs)

    if args.filter_type == 'threshold':
        threshold_filter = ent_margin_e0[dataset_name]
    elif args.filter_type == 'percentile':
        Q = 0.8
        threshold_filter = torch.kthvalue(entropy, int(Q * entropy.size(0)))[0]

    mask = entropy > threshold_filter                
    num_samples = entropy.size(0)
    max_true_count = num_samples // 2    # limit the max filter sample up to 50% batch size
    true_count = mask.sum().item()
    if true_count > max_true_count:                             
        topk_values, topk_indices = torch.topk(entropy, max_true_count)
        new_mask = torch.zeros_like(mask, dtype=torch.bool)
        new_mask[topk_indices] = True
        mask = new_mask

    filtered_entropy = entropy[mask]
    filtered_threshold = threshold_filter if args.filter_type == 'threshold' else filtered_entropy.mean()
    coff = 1 / (filtered_entropy - filtered_threshold).mean()
    return mask, coff

# Test the sample_filter function
# fused_outputs = torch.randn(10, 47)  
# print(fused_outputs)
# dataset_name = 'DTD'
# mask, coff = sample_filter(fused_outputs, dataset_name, filter_type='percentile')
# print(f"Mask: {mask}")
# print(f"Coefficient: {coff}")