data_path=                           # insert your data path here
task="DTD"                      # "SUN397,Cars,RESISC45,EuroSAT,SVHN,GTSRB,MNIST,DTD"
model_path=                       # insert your model path here
model="ViT-B-32"
model_path_fused=                    # insert your model path here. e.g.fused_model_{}_{}_{}_{}.pt
steps=200
similarity=L1                          # L1, L2, CKA
lr=4e-2
batch_size=32

python src/boost.py \
  --task ${task} \
  --data_path ${data_path} \
  --model_path ${model_path} \
  --model ${model} \
  --model_path_fused ${model_path_fused} \
  --steps ${steps} \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --similarity ${similarity}

