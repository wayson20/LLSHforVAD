0.  Install SlowFast (https://github.com/facebookresearch/SlowFast).

1.  Convert .pkl format checkpoint to .pth
    1)  Download the SlowFast Network pre-trained on Kinetics-400 dataset.
        See https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md
        "SLowFast: Kinetics/c2/SLOWFAST_8x8_R50" URL: https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/kinetics400/SLOWFAST_8x8_R50.pkl
    2)  Put the downloaded `SLOWFAST_8x8_R50.pkl` file in `pretrainings`.
        Put `configs/Kinetics/SLOWFAST_8x8_R50.yaml` (in the slowfast installation directory) in `pretrainings`.
    3)  Change directory to `1-cvt_pkl2pth`. Run `cvt_pkl2pth.sh`.
        Then `SLOWFAST_8x8_R50.pth` is generated in `pretrainings`.




