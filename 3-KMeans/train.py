import argparse
from os.path import join
from time import time as ttime

import torch

from sklearn.cluster import KMeans
from sklearn.utils import parallel_backend

from dataset import TrainingSet
from misc import get_time_stamp, get_ckpt_dir


TIME_STAMP = get_time_stamp()

parser = argparse.ArgumentParser(description='K-means for Video Anomaly Detection')
# {K}-means
parser.add_argument('--K', type=int, required=True,
                    help='{K}-means')
# Data path
parser.add_argument('--index_data', type=str, required=True,
                    help='Path to the training data (snippet-level-packaged files).')
# Others
parser.add_argument('--workers', default=24, type=int,
                    help='Number of processes.')
parser.add_argument('--note', default="", type=str,
                    help='A note for this experiment')


if __name__ == '__main__':
    args = parser.parse_args()
    KK: int = args.K

    t0 = ttime()

    # Load training data
    print(f"Loading training data ...")
    train_dataset = TrainingSet(args.index_data)
    train_mat = train_dataset.get_training_mat()

    # K-means cluster
    with parallel_backend('multiprocessing', n_jobs=args.workers):
        print(f"K={KK} clustering ... It may take a long time.")
        kmeans = KMeans(n_clusters=KK, init='k-means++', random_state=2021, n_init=10, max_iter=300, tol=0.0001)
        kmeans.fit(train_mat)

    torch.save(kmeans, join(get_ckpt_dir(TIME_STAMP, __file__), f"{KK}-means.pth"))

    t1 = ttime()
    print(f"Time={(t1-t0)/60:.1f} min")
