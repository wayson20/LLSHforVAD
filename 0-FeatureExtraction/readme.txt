0.  Install SlowFast (https://github.com/facebookresearch/SlowFast).

1.  Prepare the pre-trained model.
    1)  Download the SlowFast Network pre-trained on Kinetics-400 dataset.
        See `1-prepare_model/model_urls.txt`. Put 'SLOWFAST_8x8_R50.pkl' in `pretrainings/`.
    2)  Put `slowfast_installation_dir/configs/Kinetics/SLOWFAST_8x8_R50.yaml` in `pretrainings/`.
    3)  Convert '.pkl' format checkpoint to '.pth' file. Cd `1-prepare_model` and run `cvt_pkl2pth.sh`. 
        Then `SLOWFAST_8x8_R50.pth` is generated in `pretrainings/`.

2.  Prepare datasets.
    1)  Download Avenue, ShanghaiTech and Corridor datasets.
        See `2-prepare_data/dataset_urls.txt`.
    2)  Cd `2-prepare_data/` and run `extract_frames.py` to extract frames for videos. Use '--help' to see the usage.
        ST: only need to do this for the training videos.
        Avenue: extract frames for both training & testing videos.
        Corridor: use `mvfile_corridor.py` to put all the training/testing videos in one directory first, and then extract frames (use `--skip_first` option).
    3)  Convert the frame-level labels to '.npz' file.
        We have done this cumbersome step. The '.npz' files can be seen in `groundtruths/`.
        '.npz' file: keys: video names; values: 0 for normality, 1 for anomaly.

3.  Extract features.
    1)  ShanghaiTech & Avenue
        Cd `3-extract_features/for_ST_Avenue/`.
        Run `run.sh` to extract features for ST and Avenue datasets.
    2)  Corridor
        Cd `3-extract_features/for_Corridor/`.
        Run `recrop.py` first to extract 3 crops of each frame. Use '--help' to see the usage.
        Then run `run.sh` to extract features for Corridor dataset.
    3)  Generate snippet-level-packaged features.
        Step 1) and step 2) generated video-level-packaged features. However, snippet-level-packaged will also be used afterwards.
        Cd  `3-extract_features` and run `snippet_level_packaged.py` to convert video-level-packaged features to snippet-level-packaged features. Use '--help' to see the usage.
