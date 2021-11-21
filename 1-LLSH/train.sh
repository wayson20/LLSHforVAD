# One GPU is enough. Plese only use one GPU.
export CUDA_VISIBLE_DEVICES=0

# ST:       moco_k 8192 --batch_size 256  --iterations 60
# Avenue:   moco_k 2048 --batch_size 32   --iterations 60
# Corridor: moco_k 8192 --batch_size 256  --iterations 10

python train.py \
  "path to the **snippet**-level-packaged training data" \
  --moco_k 8192 --batch_size 256 --iterations 60 \
  --note "This is the setting for *ST* dataset."

# Two checkpoints will be saved in "../save.ckpts/1-LLSH/train_mmdd-HHMMSS/checkpoint_0|1.pth.tar"
# checkpoint_0.pth.tar: LSH; checkpoint_1.pth.tar: LLSH
