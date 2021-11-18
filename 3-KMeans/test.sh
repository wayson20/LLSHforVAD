DATA_NAME="ST"

python test.py \
  --K 32 \
  --query_data "../data.$DATA_NAME/Test" \
  --resume "../save.ckpts/3-KMeans/train_1118-163447/32-means.pth" \
  --gtnpz "../data.$DATA_NAME/gt.npz" \
  --workers 4 \
  --note "" 
