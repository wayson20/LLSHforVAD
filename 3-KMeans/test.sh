# GPUs are not used.
# --K: The number of clusters. It should be consistent with the checkpoint (--resume).

python test.py \
  --K 32 \
  --query_data "path to the **video**-level-packaged testing data" \
  --resume "path to the checkpoint, e.g., ../save.ckpts/3-KMeans/train_mmdd-HHMMSS/{--K}-means.pth" \
  --gtnpz "path to the '{dataset}_gt.npz' file" \
  --workers 4 \
  --note "" 

# The scores will be saved in "../save.results/3-KMeans/score_dict_mmdd-HHMMSS_K{--K}.pth"