export CUDA_VISIBLE_DEVICES=2

python train.py \
  "../TEMP/data.ST/smallfiles_Train" \
  --moco_k 8192 --batch_size 256 --iterations 60 \
  --note "Take a note here ..." \
  