## Codes of KNN

1.  Training\
    Calculating distances between all the training features and testing features. Run `train.sh` to calculate the (cosine) distances.\
    The distance matrices will be saved in `../save.ckpts/2-KNN/train_mmdd-HHMMSS/` dir. Note that they take a lot of disk usage (at least 2.2 TB for all the three datasets).

2.  Testing\
    Run `test.sh` for anomaly detection. The numbers of nearest neighbors are controlled by `--Ks`.\
    The scores will be saved in `../save.results/2-KNN/score_dict_mmdd-HHMMSS_{len(--Ks)}Ks.pth`.\
    All the logs will be saved in `../save.logs/2-KNN/` dir.

**For more details of training and testing settings, please refer to the scripts.**