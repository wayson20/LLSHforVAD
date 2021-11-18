export CUDA_VISIBLE_DEVICES=2

DATA_NAME="ST"

python test.py \
  --index_data "../data.$DATA_NAME/smallfiles_Train" \
  --query_data "../data.$DATA_NAME/Test" \
  --resume "../save.ckpts/1-LLSH/train_1117-220206/checkpoint_$1.pth.tar" \
  --gtnpz "../data.$DATA_NAME/gt.npz" \
  --note "" --light
