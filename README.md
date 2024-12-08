# ModelQualityBoosting

A repository of [Labeled Data Free Method for Model Quality Boosting]


## Abstract
> Large foundation models have achieved significant success in many applications, particularly in natural language processing and computer vision. Applying the large models to downstream tasks often requires fine-tuning, in order to boost the predictive accuracy on the tasks. However, the fine-tuning process needs labeled data and extensive training, which can be impractical for niche tasks like rare object detection or specialized medical applications. To address these challenges, we propose a labeled data-free method to boost model quality, by incorporating model fusion for knowledge aggregation and few-shot model specialization to the downstream tasks. Our method is powered by a carefully designed loss function considering (i) multiple granularity (i.e., task-specific and layer-specific), (ii) local and cross-task information, and (iii) mitigation of entropy minimization's error accumulation problem. Besides, our method ensures the stability of the model fusion process by cardinality-adaptive sample filtering technique with high entropy. We also design techniques to update the fused model in a few shots for specialization on downstream tasks and model quality boosting. We conduct extensive experiments on several common image classification datasets on ViT-B/32 and ViT-L/14 backbones. Our experimental results show that the proposed labeled data free method surpasses the performance of the full fine-tuning method, with performance improvements of hard image classification tasks SUN397 by 4\% and 2.3\%, based on ViT-B/32 and ViT-L/14, respectively. Moreover, our proposed model fusion method outperforms SOTA model fusion methods on multi-task learning, with average improvements by 3.1\% and 2.7\% on ViT-B/32 and ViT-L/14, respectively.


## Quick Start

### Installation
```
pip install -r requirements.txt
```

### Data acquisition

Reference: https://github.com/mlfoundations/task_vectors

### Model acquisition
[Checkpoints](https://drive.google.com/drive/folders/1t6xFYfKYpD3kUirWsIyeqjJb1liK8DDb ) for CLIP models ViT-B/32, ViT-B/16, and ViT-L/14 are now accessible via the link provided below. 

These checkpoints include fine-tuned versions specifically optimized for eight downstream tasks: Stanford Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, and SVHN.


### Execution
```
bash fuse.sh
bash boost.sh
```
## Acknowledgement
- Task Arithmetic: https://github.com/mlfoundations/task_vectors
- TIES-MERGING: https://github.com/prateeky2806/ties-merging/tree/main
- FusionBench https://github.com/tanganke/fusion_bench
- DARE: https://github.com/yule-BUAA/MergeLM
