import argparse
import tqdm
import numpy as np
import os
from os.path import join
from time import time as ttime
from typing import List

import torch
import torch.multiprocessing as mp

from dataset import TestingSet
from misc import get_time_stamp, get_logger, format_args, get_result_dir
from metrics import cal_macro_auc, cal_micro_auc


# os.environ['CUDA_VISIBLE_DEVICES'] = '2' # It's better to use `export CUDA_VISIBLE_DEVICES=` in shell.
TIME_STAMP = get_time_stamp()

parser = argparse.ArgumentParser(description='KNN for Video Anomaly Detection')
# {K}s-NN
parser.add_argument('--Ks', type=int, nargs='+', required=True,
                    help='A list of {K}s.')
# Data path
parser.add_argument('--query_data', type=str, required=True,
                    help='Path to the testing data (video-level-packaged files).')
parser.add_argument('--dist_dir', type=str, required=True,
                    help='Path to the distance dir.')
parser.add_argument('--gtnpz', type=str, required=True,
                    help='Path to groundtruth npz file.')
# Others
parser.add_argument('--workers', default=1, type=int,
                    help='Number of processes for evaluation.')
parser.add_argument('--note', default="", type=str,
                    help='A note for this experiment')


def cal_anomaly_score(i_proc: int, proc_cnt: int, score_queue: mp.Queue, args):
    '''
    Calculate anomaly scores
    '''
    test_dataset = TestingSet(root_dir=args.query_data)
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    for vid_idx in range(i_proc, len(test_dataset), proc_cnt):
        score_dict = {}
        vid_name, vid_data = test_dataset[vid_idx]

        print(f"({vid_idx+1}/{len(test_dataset)}): {vid_name}", end=' ', flush=True)

        vid_score_dict = {_K: np.zeros(len(vid_data)) for _K in args.Ks}

        n_snippets = len(test_dataset.feat_container[vid_name])
        for vid_score in vid_score_dict.values():
            assert n_snippets == vid_score.shape[0]

        _ts = ttime()
        print(f'Loading \'{vid_name}_dist.pth\'', end=' ... ', flush=True)
        dist_mat: torch.Tensor = torch.load(join(args.dist_dir, f"{vid_name}_dist.pth"))
        print(f'time={ttime()-_ts:.0f}s')

        assert len(dist_mat) % n_snippets == 0, f"{vid_name}, {n_snippets}, {len(dist_mat)}"
        dist_mat = dist_mat.view(n_snippets, len(dist_mat) // n_snippets, dist_mat.shape[1])

        for _snippet_idx in tqdm.tqdm(range(n_snippets), desc=vid_name):
            _snippet_dist = dist_mat[_snippet_idx].cuda(i_proc % n_gpus)
            _sorted_dist_val, _sorted_dist_ind = _snippet_dist.sort(dim=1)

            for _K in args.Ks:
                _candist = _sorted_dist_val[:, :_K]  # [Ncrop, K]
                _snippet_score = _candist.mean()
                vid_score_dict[_K][_snippet_idx] = _snippet_score.item()

        score_dict[vid_name] = vid_score_dict

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
    Ks: List[int] = args.Ks
    for _K in Ks:
        macro_auc = cal_macro_auc(score_dict, gt_npz, _K)
        logger.info(f"K={_K}, Macro AUC: {macro_auc*100:.1f}%")
    for _K in Ks:
        micro_auc = cal_micro_auc(score_dict, gt_npz, _K)
        logger.info(f"K={_K}, Micro AUC: {micro_auc*100:.1f}%")

    # Save scores
    torch.save(score_dict, join(get_result_dir(), f"scoredict_{TIME_STAMP}_{len(Ks)}Ks.pth"))

    t1 = ttime()
    logger.info(f"Time={(t1-t0)/60:.1f} min")
