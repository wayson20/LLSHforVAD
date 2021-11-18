0.  Install SlowFast (https://github.com/facebookresearch/SlowFast).

1.  Prepare the pre-trained model.
    1)  Download the SlowFast Network pre-trained on Kinetics-400 dataset. See `1-prepare_model/model_urls.txt`. Put it in `pretrainings/`.
    2)  Put `slowfast_installation_dir/configs/Kinetics/SLOWFAST_8x8_R50.yaml` in `pretrainings/`.
    3)  Convert '.pkl' format checkpoint to '.pth' file. Cd `1-prepare_model` and run `cvt_pkl2pth.sh`. 
        Then `SLOWFAST_8x8_R50.pth` is generated in `pretrainings/`.

2.  Prepare data
    1)  Download Avenue, ShanghaiTech and Corridor datasets. See `2-prepare_data/dataset_urls.txt`.
    2)  Cd `2-prepare_data` and run `extract_frames.py` to extract frames for videos. Use `--help` to see the usage.
        ST: only need to do this for the training videos.
        Avenue: extract frames for both training & testing videos.
        Corridor: use `mvfile_corridor.py` to put all the videos in one directory first, and then extract frames for training & testing videos (use `--skip_first` option).
    3)  Convert the groundtruth labels to '.npz' file.
        We have done this cumbersome step. The '.npz' files can be seen in `groundtruths/`.

3. 
