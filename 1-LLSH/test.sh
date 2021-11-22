# One GPU is enough. Plese only use one GPU.
export CUDA_VISIBLE_DEVICES=0

# Resume the checkpoint saved in "../save.ckpts/1-LLSH/train_mmdd-HHMMSS/checkpoint_0|1.pth.tar". Note that '--moco_k' should be consistent with the checkpoint.
# checkpoint_0.pth.tar: LSH; checkpoint_1.pth.tar: LLSH
# Append '--light' option to use light-LLSH.

python test.py \
  --index_data "path to the **snippet**-level-packaged training data" \
  --query_data "path to the **video**-level-packaged testing data" \
  --resume "path to the checkpoint, e.g. ../save.ckpts/train_mmdd-HHMMSS/checkpoint_0|1.pth.tar" \
  --moco_k 8192 \
  --gtnpz "path to the '{dataset}_gt.npz' file" \
  --note "" # --light

# The testing models and scores will be saved in "../save.results/1-LLSH/"
