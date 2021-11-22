# GPUs are not used.
# --K: The number of clusters. The calculation cost will almost increase linearly with the grouth of K.

python train.py \
  --K 32 \
  --index_data "path to the **snippet**-level-packaged training data" \
  --workers 32 \
  --note ""

# The KMeans model will be saved in "../save.ckpts/3-KMeans/train_mmdd-HHMMSS/{--K}-means.pth"
