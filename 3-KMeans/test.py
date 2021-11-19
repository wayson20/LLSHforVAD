import argparse
from os.path import join
from time import time as ttime

import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.multiprocessing as mp

from dataset import TestingSet
from metrics import cal_macro_auc, cal_micro_auc
from misc import get_time_stamp, get_logger, get_result_dir, format_args


TIME_STAMP = get_time_stamp()

parser = argparse.ArgumentParser(description='K-means for Video Anomaly Detection')
# {K}-means
parser.add_argument('--K', type=int, required=True,
                    help='{K}-means')
# Data path
parser.add_argument('--query_data', type=str, required=True,
                    help='Path to the training data (snippet-level-packaged files).')
parser.add_argument('--resume', type=str, required=True,
                    help='Path to a checkpoint.')
parser.add_argument('--gtnpz', type=str, required=True,
                    help='Path to groundtruth npz file.')
# Others
parser.add_argument('--workers', default=4, type=int,
                    help='Number of processes for evaluation.')
parser.add_argument('--note', default="", type=str,
                    help='A note for this experiment')


def cal_anomaly_score(i_proc: int, proc_cnt: int, score_queue: mp.Queue, args):
    '''
    Calculate anomaly scores
    '''
    kmeans: KMeans = torch.load(args.resume)
    k_centers: torch.Tensor = torch.from_numpy(kmeans.cluster_centers_)  # [K, ddim]
    assert args.K == len(k_centers), f"args.K({args.K}) != k_centers({len(k_centers)})"

    test_dataset = TestingSet(root_dir=args.query_data)
    Ncrop = len(test_dataset[0][1][0])

    k_centers = k_centers.unsqueeze(1).repeat(1, Ncrop, 1)  # [K, Ncrop, ddim]

    for vid_idx in range(i_proc, len(test_dataset), proc_cnt):
        vid_name, vid_data = test_dataset[vid_idx]

        print(f"({vid_idx+1}/{len(test_dataset)}): {vid_name}")

        n_snippets = len(vid_data)
        vid_scores: np.ndarray = np.zeros(n_snippets)

        score_dict = {}
        for _snippet_idx in range(n_snippets):
            _snippet_data: torch.Tensor = vid_data[_snippet_idx]  # [Ncrop, T, H, W, C]
            _snippet_data = _snippet_data.flatten(1)  # [Ncrop, ddim]
            _snippet_data = _snippet_data.unsqueeze(0).repeat(args.K, 1, 1)  # [1, Ncrop, ddim]

            _candist: torch.Tensor = torch.norm(_snippet_data - k_centers, p=2, dim=-1)  # [K, Ncrop]
            _snippet_score = _candist.amin(0).mean()

            vid_scores[_snippet_idx] = _snippet_score.item()

        score_dict[vid_name] = vid_scores

        assert not score_queue.full()
        score_queue.put(score_dict)


if __name__ == '__main__':
    args = parser.parse_args()

    logger = get_logger(TIME_STAMP, __file__)
    logger.info(format_args(args))

    t0 = ttime()

    gt_npz = np.load(args.gtnpz)

    # Calculate anomaly scores
    len_dataset = len(TestingSet(root_dir=args.query_data))
    score_queue = mp.Manager().Queue(maxsize=len_dataset)

    mp.spawn(cal_anomaly_score, args=(args.workers, score_queue, args), nprocs=args.workers)

    assert score_queue.full()
    score_dict = {}
    while not score_queue.empty():
        score_dict.update(score_queue.get())
    assert len(score_dict) == len_dataset

    # Calculate AUC
    macro_auc = cal_macro_auc(score_dict, gt_npz)
    micro_auc = cal_micro_auc(score_dict, gt_npz)
    logger.info(f"Macro AUC: {macro_auc*100:.1f}%")
    logger.info(f"Micro AUC: {micro_auc*100:.1f}%")

    # Save scores
    torch.save(score_dict, join(get_result_dir(), f"scoredict_{TIME_STAMP}_K{args.K}.pth"))

    t1 = ttime()
    logger.info(f"Time={(t1-t0)/60:.1f} min")
