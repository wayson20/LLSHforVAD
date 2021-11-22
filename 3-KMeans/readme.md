## Codes of K-means

1.  Training\
    Run `train.sh` for training the KMeans model. `--K` denotes the number of clusters.\
    It will take a long time for large `--K`.\
    The KMeans model will be saved in `../save.ckpts/3-KMeans/train_mmdd-HHMMSS/{--K}-means.pth`.

2.  Testing\
    Run `test.sh` for anomaly detection. `--K` should be consistent with the resumed checkpoint.\
    The scores will be saved in `../save.results/3-KMeans/score_dict_mmdd-HHMMSS_K{--K}.pth`.
    All the logs will be saved in `../save.logs/3-KMeans/` dir.

**For more details of training and testing settings, please refer to the scripts.**