export CUDA_VISIBLE_DEVICES=2
DATA_NAME='ST'

python train.py \
  --index_data "../TEMP/data.$DATA_NAME/smallfiles_Train" \
  --query_data "../TEMP/data.$DATA_NAME/smallfiles_Test" \
  --workers 1 \
  --note ""
  