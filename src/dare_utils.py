import torch
import sys
import re
import task_vector
sys.path.append('.')

# task_vectors_pruned = mask_model_weights(task_vectors, mask_rate=0.9, mask_strategy='magnitude')
def mask_model_weights(task_vectors: list, exclude_param_names_regex: list, use_weight_rescale: bool=True, mask_strategy: str='magnitude', mask_rate: float=0.5, mask_num:int=0):
    
    task_vectors_pruned = []
    
    for task_vector in task_vectors:
        task_vector_pruned = {}
        masked_param_dict = {}
        # for key in task_vector.vector:
            # task_vector_pruned[key] = mask_input_with_mask_rate(task_vector.vector[key], mask_rate=mask_rate, use_rescale=use_weight_rescale, mask_strategy=mask_strategy)
        # task_vector.vector = task_vector_pruned
        
        model_param_dict = task_vector.vector
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(model_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        
        # param_dict = {param_name: param_value for param_name, param_value in model_param_dict}
        model_param_dict = {param_name: model_param_dict[param_name] for param_name in param_names_to_merge}
        
        for param_name, param_value in model_param_dict.items():
            masked_param_dict[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy,mask_num=mask_num)
        task_vector.vector = masked_param_dict
        task_vectors_pruned.append(task_vector)
    
    return task_vectors_pruned
def mask_model_weights_ties(task_vectors: list, exclude_param_names_regex: list, use_weight_rescale: bool=True, mask_strategy: str='magnitude', mask_rate: float=0.5, mask_num:int=0):
    
    task_vectors_pruned = []
    
    for task_vector in task_vectors:
        task_vector_pruned = {}
        masked_param_dict = {}
        
        # model_param_dict = task_vector.vector
        model_param_dict = task_vector

        param_names_to_merge = get_param_names_to_merge(input_param_names=list(model_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        
        model_param_dict = {param_name: model_param_dict[param_name] for param_name in param_names_to_merge}
        
        for param_name, param_value in model_param_dict.items():
            masked_param_dict[param_name] = mask_input_with_mask_rate(input_tensor=param_value, mask_rate=mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy,mask_num=mask_num)
        # task_vector.vector = masked_param_dict
        # task_vectors_pruned.append(task_vector)
        task_vectors_pruned.append(masked_param_dict)
    
    return task_vectors_pruned

def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str,mask_num:int):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    elif mask_strategy == "magnitude":
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        len_input_tensor = len(input_tensor)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)                 # Last smallest k values (num_mask_params)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)

    elif mask_strategy == "topk":
        assert mask_strategy == "topk", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = mask_num
        len_input_tensor = len(input_tensor)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)                 # Last smallest k values (num_mask_params)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
        
        
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
        
        
    return masked_input_tensor

def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def dare(task_vectors:list, mask_rate=0.5, rescale_flag=True, mask_strategy='random'):
    exclude_param_names_regex = [".*classifier.*", ".*pooler.*", 
                             ".*LayerNorm.*", ".*bias.*", ".*gamma.*", 
                             ".*beta.*", ".*position_ids.*", ".*token_type_ids.*", 
                             ".*word_embeddings.*", ".*position_embedding.*", ".*token_type_embeddings.*", 
                             ".*pooler.*", ".*LayerNorm.*", ".*bias.*", ".*gamma.*", ".*beta.*", ".*position_ids.*", 
                             ".*token_type_ids.*", ".*word_embeddings.*", ".*position_embeddings.*", ".*token_type_embeddings.*",
                             ".positional_embedding*", 
                             "text_projection*",
                             ".logit_scale*",    # must be excluded
                             "class_embedding*",
                             "posional_embedding*",
                             ".proj",
                             ]
    exclude_param_names_regex = [".*logit_scale*"]
    exclude_param_names_regex=[".*classifier.*"]
    task_vectors_pruned = mask_model_weights(task_vectors, mask_rate=mask_rate, use_weight_rescale=rescale_flag, mask_strategy=mask_strategy,exclude_param_names_regex=exclude_param_names_regex)
    return task_vectors_pruned

def vector_recovery(merged_vector: task_vector.TaskVector, task_vectors: list, exclude_param_names_regex: list):
        
    for task_vector in task_vectors:
        model_param_dict = task_vector.vector
        merged_param = get_param_names_to_merge(input_param_names=list(merged_vector.vector.keys()), exclude_param_names_regex=exclude_param_names_regex)
        merged_param_dict = {param_name: model_param_dict[param_name] for param_name in merged_param}
        
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(model_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = {param_name: model_param_dict[param_name] for param_name in param_names_to_merge}
        for key in model_param_dict:
            if model_param_dict[key].dtype in [torch.int64, torch.uint8]:           # ?????
                continue
            merged_param_dict[key] += model_param_dict[key] - merged_param_dict[key]
        # print(merged_param_dict['model.visual.transformer.resblocks.8.mlp.c_proj.bias'][1])
        merged_vector.vector = merged_param_dict  
    
    return merged_vector