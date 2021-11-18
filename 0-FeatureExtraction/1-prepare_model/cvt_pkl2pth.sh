export CUDA_VISIBLE_DEVICES="0"

python cvt_pkl2pth.py \
  --cfg "../pretrainings/SLOWFAST_8x8_R50.yaml" \
  TEST.CHECKPOINT_FILE_PATH "../pretrainings/SLOWFAST_8x8_R50.pkl" \
  TEST.CHECKPOINT_TYPE "caffe2" \
  TRAIN.ENABLE False NUM_GPUS 1
