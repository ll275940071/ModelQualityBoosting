# ModelQualityBoosting

A repository of [Labeled Data Free Method for Model Quality Boosting]


## Abstract
> Large foundation models have achieved significant success in many applications, particularly in natural language processing and computer vision. Applying the large models to downstream tasks often requires fine-tuning, in order to boost the predictive accuracy on the tasks. However, the fine-tuning process needs labeled data and extensive training, which can be impractical for niche tasks like rare object detection or specialized medical applications. To address these challenges, we propose a labeled data-free method to boost model quality, by incorporating model fusion for knowledge aggregation and few-shot model specialization to the downstream tasks. Our method is powered by a carefully designed loss function considering (i) multiple granularity (i.e., task-specific and layer-specific), (ii) local and cross-task information, and (iii) mitigation of entropy minimization's error accumulation problem. Besides, our method ensures the stability of the model fusion process by cardinality-adaptive sample filtering technique with high entropy. We also design techniques to update the fused model in a few shots for specialization on downstream tasks and model quality boosting. We conduct extensive experiments on several common image classification datasets on ViT-B/32 and ViT-L/14 backbones. Our experimental results show that the proposed labeled data free method surpasses the performance of the full fine-tuning method, with performance improvements of hard image classification tasks SUN397 by 4\% and 2.3\%, based on ViT-B/32 and ViT-L/14, respectively. Moreover, our proposed model fusion method outperforms SOTA model fusion methods on multi-task learning, with average improvements by 3.1\% and 2.7\% on ViT-B/32 and ViT-L/14, respectively.


## Quick Start

### Installation
```
pip install -r requirements.txt
```

### Quick Example
```
bash fuse.sh
```
