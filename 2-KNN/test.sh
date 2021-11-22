export CUDA_VISIBLE_DEVICES=0

# Ks: {K}-NN
# --workers: We recommend setting it to 1 because more processes are difficult to speed up reading from the hard disk.

python test.py \
  --Ks 1 32 64 128 256 512 1024 2048 \
  --query_data "path to the **video**-level-packaged testing data" \
  --dist_dir "path to the saved distance matrices, e.g., ./save.ckpts/2-KNN/train_mmdd-HHMMSS/" \
  --gtnpz "path to the '{dataset}_gt.npz' file" \
  --workers 1 \
  --note "" 

# The scores will be saved in "../save.results/2-KNN/score_dict_mmdd-HHMMSS_{len(--Ks)}Ks.pth"
