export CUDA_VISIBLE_DEVICES=2

DATA_NAME="ST"

python test.py \
  --Ks 1 32 64 128 256 512 1024 2048 \
  --query_data "../data.$DATA_NAME/Test" \
  --dist_dir "../save.ckpts/2-KNN/train_1117-231535" \
  --gtnpz "../data.$DATA_NAME/gt.npz" \
  --workers 1 \
  --note "" 
