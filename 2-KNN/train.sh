export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
  --index_data "path to the **snippet**-level-packaged training data" \
  --query_data "path to the **snippet**-level-packaged testing data" \
  --workers 4 \
  --note ""
  
# Distance matrices will be saved in "../save.ckpts/2-KNN/train_mmdd-HHMMSS".
# Note that the distance matrices will take a huge disk usage (7.3 GB for Avenue, 333 GB for ST, 1.8 TB for Corridor).
