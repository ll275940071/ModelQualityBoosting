import torch
from aux import *   
from heads import get_classification_head

class CustomModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        super(CustomModelWrapper, self).__init__()
        self.model = model
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')
    def forward(self, images):
        features = self.model(images)
        return features

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)
def functionalize(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names
# def functionalize(mod):
#     orig_params = []
#     names = []
#     for name, param in mod.named_parameters():
#         orig_params.append(param.detach().clone())
#         del_attr(mod, name.split("."))
#         names.append(name)
#     return orig_params, names
def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class FusedModel(torch.nn.Module):
    def __init__(self, args, ptm_model, paramslist, names):
        super(FusedModel, self).__init__()
        self.paramslist = paramslist
        self.model = ptm_model
        self.names = names
        prior = 0.3
        self.granularity = args.granularity
        if args.granularity=='task':                                        # task granularity
            self.pretrain_lambdas = torch.ones(1, 1)
            rlambdas = torch.ones(1, len(paramslist)-1) * prior
            print('**** Begin fusion with task granularity ****')
        elif args.granularity=='layer':                                                               # layer granularity
            self.pretrain_lambdas = torch.ones(len(paramslist[0]), 1)
            rlambdas = torch.ones(len(paramslist[0]), len(paramslist)-1) * prior  # (1 * tasks)
            print('**** Begin fusion with layer granularity ****')
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        self.classifier = []
        for dataset_name in args.datasets:
            classification_head = get_classification_head(args, dataset_name)
            layer_name = 'classifier_{}'.format(dataset_name)
            self.add_module(layer_name, classification_head.to(args.device))
            self.classifier.append(layer_name)

    def lambdas(self):
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        return [self.lambdas_raw]
    
    def count_trainable_params(self):
        trainable_params = self.collect_trainable_params()
        total_params = sum(p.numel() for p in trainable_params)
        print(f'Total number of trainable parameters: {total_params}')
        return total_params
    
    def get_classification_head(self, dataset_name):
        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        alph = self.lambdas()
        if self.granularity == 'task':
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        else:
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model
        
    def forward(self, inp, dataset_name):
        alph = self.lambdas()
        if self.granularity == 'task':
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        else:
            params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = 'classifier_{}'.format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)

        return out