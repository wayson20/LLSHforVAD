DATA_NAME='ST'

python train.py \
  --K 32 \
  --index_data "../data.$DATA_NAME/smallfiles_Train" \
  --workers 32 \
  --note ""
  