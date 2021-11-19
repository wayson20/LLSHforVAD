import argparse
import os
from os.path import join
from math import ceil
from time import time as ttime
import tqdm

import torch
import torch.multiprocessing as mp
torch.set_num_threads(48)

from dataset import TrainingSet
from misc import get_time_stamp, get_ckpt_dir


# os.environ['CUDA_VISIBLE_DEVICES'] = '2' # It's better to use `export CUDA_VISIBLE_DEVICES=` in shell.
TIME_STAMP = get_time_stamp()

parser = argparse.ArgumentParser(description='KNN for Video Anomaly Detection')
# Data path
parser.add_argument('--index_data', type=str, required=True,
                    help='Path to the training data (snippet-level-packaged files).')
parser.add_argument('--query_data', type=str, required=True,
                    help='Path to the testing data (snippet-level-packaged files).')
# Others
parser.add_argument('--workers', default=4, type=int,
                    help='Number of processes for evaluation.')
parser.add_argument('--note', default="", type=str,
                    help='A note for this experiment')


def cossim_dist(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    '''
    X: (N, ddim)
    Y: (M, ddim)
    return dist_mat: (N,M)
    '''
    X = X / X.norm(2, dim=-1, keepdim=True)
    Y = Y / Y.norm(2, dim=-1, keepdim=True)
    dist_mat = 1 - torch.matmul(X, Y.T)
    return dist_mat


def cossim_dist_gpu(X: torch.Tensor, Y: torch.Tensor, i_gpu: int, split_size: int) -> torch.Tensor:
    '''
    X: (N, ddim)
    Y: (M, ddim)
    return dist_mat: (N,M)
    '''
    Ys = Y.split(ceil(Y.shape[0] / split_size))
    dist_mat = []
    for _Y in Ys:
        _Y: torch.Tensor
        _d = cossim_dist(X.cuda(i_gpu), _Y.cuda(i_gpu)).cpu()
        dist_mat.append(_d)
    return torch.cat(dist_mat, 1)


def cal_dist(i_proc: int, proc_cnt: int, train_mat_pth, args, dist_dir):
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    train_mat = torch.load(train_mat_pth)

    vid_name_list = sorted(os.listdir(args.query_data))

    for vid_idx in range(i_proc, len(vid_name_list), proc_cnt):
        vid_name = vid_name_list[vid_idx]
        print(f"({vid_idx+1}/{len(vid_name_list)}): {vid_name}")

        vid_dir = join(args.query_data, vid_name)
        test_mat = []

        for _snippet_name in sorted(os.listdir(vid_dir)):
            _snippet_data: torch.Tensor = torch.load(join(vid_dir, _snippet_name))
            test_mat.append(_snippet_data.flatten(1))

        test_mat = torch.cat(test_mat, 0)
        dist_mat = cossim_dist_gpu(test_mat, train_mat, i_proc % n_gpus, 8)

        torch.save(dist_mat, join(dist_dir, f"{vid_name}_dist.pth"))


if __name__ == '__main__':
    args = parser.parse_args()
    proc_cnt: int = args.workers

    dist_dir = get_ckpt_dir(TIME_STAMP, __file__)

    t0 = ttime()

    # Load training data
    print("Loading training data ...")
    train_dataset = TrainingSet(args.index_data)
    train_mat = train_dataset.get_training_mat()
    _train_mat_pth = f"temp_{TIME_STAMP}.pth"
    torch.save(train_mat, _train_mat_pth)

    # Calculate distances
    mp.spawn(cal_dist, args=(proc_cnt, _train_mat_pth, args, dist_dir), nprocs=proc_cnt)
    os.remove(_train_mat_pth)

    t1 = ttime()
    print(f"Time={(t1-t0)/60:.1f} min")
