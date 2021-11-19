# This is a script for extracting features of Corridor dataset.
# Please run `recrop.py` first to extract 3 crops of each frame, and then run this script using the offline cropped images.
# 'DATA.PATH_TO_DATA_DIR' should be set as the cropped images dir.

export CUDA_VISIBLE_DEVICES=0,1,2,3

python main.py \
  --cfg "../../pretrainings/SLOWFAST_8x8_R50.yaml" \
  TEST.CHECKPOINT_FILE_PATH "../../pretrainings/SLOWFAST_8x8_R50.pth" \
  DATA.PATH_TO_DATA_DIR "** the dir containing extracted frames (images extracted by `recrop.py`), e.g., 'DATA.PATH_TO_DATA_DIR'/209/[000000_1.jpg, 000000_2.jpg, 000000_3.jpg ...] **" \
  TEST.SAVE_RESULTS_PATH "** the dir to save extracted features, e.g., 'TEST.SAVE_RESULTS_PATH'/([209.pth, 210.pth, ...]) **" \
  DATA.NUM_FRAMES 32 DATA.SAMPLING_RATE 1 \
  DATA.PATH_PREFIX "** length of the frame name, e.g., '000000_1.jpg': 'DATA.PATH_PREFIX'='XXXXXX' (use 'X' plachholders) **" \
  DATA_LOADER.NUM_WORKERS 4