data_path=/home/xuyuebin/Documents/datasets/merging_dataset
datasets="SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD"
model="ViT-B-32"
model_path=/home/xuyuebin/Documents/models/
granularity="task"    # task, layer
fuse_backbone="TA"  # TA(task_arithmetic), Ties, DARE
epoch=200
batch_size=8
lr=1e-3
dare="False"  # True, False
EM="False"  # True, False

cmd="python src/fuse.py \
  --datasets ${datasets} \
  --data_path ${data_path} \
  --model ${model} \
  --model_path ${model_path} \
  --granularity ${granularity} \
  --fuse_backbone ${fuse_backbone}"

if [ "$dare" = "True" ]; then
  cmd="$cmd --dare"
  cmd="$cmd --mask_rate 0.5"
  cmd="$cmd --rescale_flag"
  cmd="$cmd --mask_strategy random"
fi

eval $cmd