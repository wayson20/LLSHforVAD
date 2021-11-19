import argparse
import numpy as np
from collections import OrderedDict
import tqdm
import os
from os import listdir
from os.path import join, basename, isfile
from time import time as ttime
from typing import Union

import torch
import torch.cuda
import torch.multiprocessing as mp

from moco import MoCo
from network import HashNet
from dataset import TestingSet
from llsh import LLSH
from lightllsh import LightLLSH

from misc import get_time_stamp, get_logger, get_result_dir, format_args
from metrics import cal_macro_auc, cal_micro_auc


# os.environ['CUDA_VISIBLE_DEVICES'] = '2' # It's better to use `export CUDA_VISIBLE_DEVICES=` in shell.
TIME_STAMP = get_time_stamp()

parser = argparse.ArgumentParser(description='Learnable Locality-Sensitive Hashing for Video Anomaly Detection')
# Data path
parser.add_argument('--index_data', type=str, required=True,
                    help='Path to the training data (snippet-level-packaged files).')
parser.add_argument('--query_data', type=str, required=True,
                    help='Path to the testing data (video-level-packaged files).')
parser.add_argument('--resume', type=str, required=True,
                    help='Path to a checkpoint')
parser.add_argument('--gtnpz', type=str, required=True,
                    help='Path to groundtruth npz file.')
# Use light-LLSH
parser.add_argument('--light', action='store_true',
                    help='Use light-LLSH.')
# Hash encoder settingts
parser.add_argument('--len_hash_code', default=32, type=int, choices=(32,),
                    help='Length of hash codes, i.e., r')
parser.add_argument('--num_hash_layer', default=8, type=int, choices=(8,),
                    help='Number of hash layers, i.e., b')
# MoCo settings
parser.add_argument('--moco_k', default=8192, type=int, choices=(8192, 2048),
                    help='Length of the queue, i.e., l (ST: 8192; Avenue: 2048; Corridor: 8192)' +
                    'Actually, it is not used in testing phase.')
parser.add_argument('--moco_m', default=0.999, type=float, choices=(0.999,),
                    help="Momentum of updating hash encoder E_k (in the right part), i.e., m" +
                    'Actually, it is not used in testing phase.')
parser.add_argument('--moco_t', default=0.2, type=float, choices=(0.2,),
                    help='Temperature for InfoNCE loss, i.e., tau' +
                    'Actually, it is not used in testing phase.')
# Others
parser.add_argument('--workers', default=4, type=int,
                    help='Number of processes for evaluation.')
parser.add_argument('--note', default="", type=str,
                    help='A note for this experiment')
parser.add_argument('--print_model', action='store_true',
                    help='Print the model.')


def load_model(args) -> torch.nn.Module:
    model = MoCo(moco_K=args.moco_k,
                 moco_m=args.moco_m,
                 moco_T=args.moco_t,
                 queue_dim=args.len_hash_code * args.num_hash_layer,
                 Network=HashNet,
                 feat_dim=[1, 2, 2, 2304],
                 len_hash_code=args.len_hash_code,
                 num_hash_layer=args.num_hash_layer,
                 )

    if args.resume:
        if isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location='cpu')

            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k[len('module.'):]] = v

            model.load_state_dict(new_state_dict)

            del checkpoint
        else:
            raise FileNotFoundError("No checkpoint found at '{}'".format(args.resume))
    else:
        raise NotImplementedError("A checkpoint should be loaded.")

    encoder_q = model.encoder_q
    encoder_q.eval()

    del model

    if args.print_model:
        print(encoder_q)

    return encoder_q


def cal_anomaly_score(i_proc: int, proc_cnt: int, score_queue: mp.Queue, args, llsh_pth: str):
    '''
    Calculate anomaly scores 
    '''
    llsh_inst: Union[LLSH, LightLLSH] = torch.load(llsh_pth)

    test_dataset = TestingSet(root_dir=args.query_data)

    for vid_idx in range(i_proc, len(test_dataset), proc_cnt):
        vid_name, vid_data = test_dataset[vid_idx]

        print(f"({vid_idx+1}/{len(test_dataset)}): {vid_name}")
        vid_scores: np.ndarray = np.zeros(len(vid_data))

        n_snippets = len(test_dataset.feat_container[vid_name])
        assert n_snippets == vid_scores.shape[0]

        score_dict = {}
        for _snippet_idx in range(n_snippets):
            _snippet_data: torch.Tensor = vid_data[_snippet_idx]  # [Ncrop, T, H, W, C]

            _candist: torch.Tensor = llsh_inst.batch_query(_snippet_data)  # [Ncrop, b]
            _snippet_score = _candist.amin(1).mean()
            vid_scores[_snippet_idx] = _snippet_score.item()

        score_dict[vid_name] = vid_scores

        assert not score_queue.full()
        score_queue.put(score_dict)


if __name__ == '__main__':
    args = parser.parse_args()

    logger = get_logger(TIME_STAMP, __file__)
    logger.info(format_args(args))

    if torch.cuda.device_count() != 1:
        logger.warn("Please use only one GPU.")
        exit()

    t0 = ttime()

    gt_npz = np.load(args.gtnpz)

    # Load model
    _hash_net: HashNet = load_model(args)
    llsh_inst = None
    if args.light:
        llsh_inst = LightLLSH(_hash_net)
    else:
        llsh_inst = LLSH(_hash_net)

    # Index stage
    print("Index stage ...")
    for _vid_name in tqdm.tqdm(sorted(listdir(args.index_data)), desc='Index stage'):
        vid_dir = join(args.index_data, _vid_name)
        train_mat = []
        for _snippet_name in sorted(listdir(vid_dir)):
            snippet_pth: torch.Tensor = torch.load(join(vid_dir, _snippet_name))
            train_mat.append(snippet_pth.flatten(1))
        train_mat = torch.cat(train_mat, 0)
        llsh_inst.batch_index(train_mat)
    llsh_inst.cvt_list2tensor()

    # Save the model temporarily
    llsh_pth = join(get_result_dir(), f"llsh_inst_{TIME_STAMP}.pth")
    torch.save(llsh_inst, llsh_pth)
    del llsh_inst
    torch.cuda.empty_cache()

    # Query stage
    print("Query stage ...")
    len_dataset = len(TestingSet(root_dir=args.query_data))
    score_queue = mp.Manager().Queue(maxsize=len_dataset)

    mp.spawn(cal_anomaly_score, args=(args.workers, score_queue, args, llsh_pth), nprocs=args.workers)

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
    epoch_name: str = basename(args.resume).split('_')[1].split('.')[0]
    np.savez(join(get_result_dir(), f"score_dict_{TIME_STAMP}_{epoch_name}.npz"), **score_dict)

    t1 = ttime()
    logger.info(f"Time={(t1-t0)/60:.1f} min")
