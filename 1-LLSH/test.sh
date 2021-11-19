export CUDA_VISIBLE_DEVICES=2

DATA_NAME="ST"

python test.py \
  --index_data "../TEMP/data.$DATA_NAME/smallfiles_Train" \
  --query_data "../TEMP/data.$DATA_NAME/Test" \
  --resume "../save.ckpts/1-LLSH/train_1119-212006/checkpoint_$1.pth.tar" \
  --gtnpz "../TEMP/data.$DATA_NAME/gt.npz" \
  --note ""
