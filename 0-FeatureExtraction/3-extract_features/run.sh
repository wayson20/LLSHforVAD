export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py \
  --cfg "../pretrainings/SLOWFAST_8x8_R50.yaml" \
  TEST.CHECKPOINT_FILE_PATH "../pretrainings/SLOWFAST_8x8_R50.pth" \
  DATA.PATH_TO_DATA_DIR "** the dir containing extracted frames, e.g., 'DATA.PATH_TO_DATA_DIR'/01_001/[000000.jpg, ...] **" \
  TEST.SAVE_RESULTS_PATH "** the dir to save extracted features, e.g., 'TEST.SAVE_RESULTS_PATH'/[01_001.pth, 01_002.pth, ...] **" \
  DATA.NUM_FRAMES 32 DATA.SAMPLING_RATE 1 \
  DATA.PATH_PREFIX "** length of the frame name, e.g., '000000.jpg': DATA.PATH_PREFIX='XXXXXX' (use 'X' plachholders) **" \
  DATA_LOADER.NUM_WORKERS 4