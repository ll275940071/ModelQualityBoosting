data_path=/home/xuyuebin/Documents/datasets/merging_dataset
target_datasets="Cars"
model="ViT-B-32"
model_path=/home/xuyuebin/Documents/models/
steps=100
batch_size=16
lr=1e-3

python src/finetune.py \
  --data_path ${data_path} \
  --target_datasets ${target_datasets} \
  --model ${model} \
  --model_path ${model_path} \
  --steps ${steps} \
  --batch_size ${batch_size} \
  --lr ${lr}

