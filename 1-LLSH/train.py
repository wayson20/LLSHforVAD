import argparse
import builtins
import time
import os

import torch
import random
import numpy as np

rand_seed = 2021
random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)

import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data

from moco import MoCo
from network import HashNet
from dataset import TrainingSet

from misc import get_time_stamp, get_logger, format_args, get_ckpt_dir
from misc import AverageMeter, ProgressMeter, accuracy, save_checkpoint


# os.environ['CUDA_VISIBLE_DEVICES'] = '2' # It's better to use `export CUDA_VISIBLE_DEVICES=` in shell.
TIME_STAMP = get_time_stamp()

parser = argparse.ArgumentParser(description='Learnable Locality-Sensitive Hashing for Video Anomaly Detection')
# Data path
parser.add_argument('data', type=str,
                    help='Path to the training data (snippet-level-packaged files).')
# Common training settings
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, choices=(0.001,),
                    help='Learning rate.')
parser.add_argument('--epochs', default=1, type=int, choices=(0, 1),
                    help='Number of total epochs to run.')
parser.add_argument('--iterations', default=60, type=int, choices=(60,),
                    help='A way to simulate more epochs. (ST: 60; Avenue: 60; Corridor: 10)')
parser.add_argument('--batch_size', default=256, type=int, choices=(256, 32),
                    help='Batch size. (ST: 256; Avenue: 32; Corridor: 256)')
parser.add_argument('--workers', default=16, type=int, choices=(16,),
                    help='Number of data loading workers')
parser.add_argument('--start-epoch', default=0, type=int, choices=(0,),
                    help='Manual epoch number.')
# Optimizer settings
parser.add_argument('--momentum', default=0.9, type=float, choices=(0.9,),
                    help='Momentum of SGD pptimizer.')
parser.add_argument('--weight_decay', default=1e-4, type=float, choices=(1e-4,),
                    help='Weight decay.')
parser.add_argument('--schedule', default=[999], nargs='*', type=int, choices=(999,),
                    help='Learning rate schedule.')
# Distributed settings
parser.add_argument('--world_size', default=1, type=int, choices=(1,),
                    help='Number of nodes for distributed training.')
parser.add_argument('--rank', default=0, type=int, choices=(0,),
                    help='Node rank for distributed training.')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:10000', type=str,
                    help='Url used to set up distributed training. Change the port if it is occupied')
parser.add_argument('--dist_backend', default='nccl', type=str, choices=('nccl',),
                    help='Distributed backend.')
parser.add_argument('--mp_distributed', default=True, type=bool, choices=(True,),
                    help='Use multiprocessing or not.')
# Dataset settings:
parser.add_argument('--t_rand_range', default=150, type=int, choices=(150,),
                    help="Sampling offset, i.e., deltaVar_t")
# Hash encoder settingts
parser.add_argument('--len_hash_code', default=32, type=int, choices=(32,),
                    help='Length of hash codes, i.e., r')
parser.add_argument('--num_hash_layer', default=8, type=int, choices=(8,),
                    help='Number of hash layers, i.e., b')
# MoCo settings
parser.add_argument('--moco_k', default=8192, type=int, choices=(8192, 2048),
                    help='Length of the queue, i.e., l (ST: 8192; Avenue: 2048; Corridor: 8192)')
parser.add_argument('--moco_m', default=0.999, type=float, choices=(0.999,),
                    help="Momentum of updating hash encoder E_k (in the right part), i.e., m")
parser.add_argument('--moco_t', default=0.2, type=float, choices=(0.2,),
                    help='Temperature for InfoNCE loss, i.e., tau')
# Others
parser.add_argument('--note', default="", type=str,
                    help='A note for this experiment')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    help='Print frequency.')
parser.add_argument('--print_model', action='store_true',
                    help='Print the model.')


def main():
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.mp_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node != 1:
        print("Please use only one GPU.")
        exit()

    if args.mp_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        raise NotImplementedError("Please set '--mp_distributed=True'")


def main_worker(proc_id, ngpus_per_node, args):
    '''
    Main training process.
    '''
    gpu_id = proc_id % ngpus_per_node

    if args.mp_distributed and proc_id != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        assert proc_id == 0
        logger = get_logger(TIME_STAMP, __file__)
        builtins.print = logger.info

    print(format_args(args))

    if args.distributed:
        if args.mp_distributed:
            args.rank = args.rank * ngpus_per_node + proc_id
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = MoCo(moco_K=args.moco_k,
                 moco_m=args.moco_m,
                 moco_T=args.moco_t,
                 queue_dim=args.len_hash_code * args.num_hash_layer,
                 Network=HashNet,
                 feat_dim=[1, 2, 2, 2304],
                 len_hash_code=args.len_hash_code,
                 num_hash_layer=args.num_hash_layer,
                 )

    if args.print_model:
        print(model)

    if args.distributed:
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])
    else:
        raise NotImplementedError()
    criterion = nn.CrossEntropyLoss().cuda(gpu_id)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    train_dataset = TrainingSet(args.data, args.t_rand_range, args.iterations)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True,
        sampler=train_sampler, drop_last=True, prefetch_factor=1)

    def save_epoch(epoch):
        if not args.mp_distributed or (args.mp_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            },
                is_best=False,
                filedir=get_ckpt_dir(TIME_STAMP, __file__),
                epoch=epoch)

    save_epoch(0)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, gpu_id, args)

        save_epoch(epoch + 1)


def train(train_loader, model, criterion, optimizer, gpu_id, args):
    '''
    Training for one epoch.
    '''
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Iteration: ")

    model.train()

    end = time.time()
    for i, snippets in enumerate(train_loader):
        data_time.update(time.time() - end)

        if gpu_id is not None:
            snippets[0] = snippets[0].cuda(gpu_id, non_blocking=True)
            snippets[1] = snippets[1].cuda(gpu_id, non_blocking=True)

        output, target = model(data_q=snippets[0], data_k=snippets[1])
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), snippets[0].size(0))
        top1.update(acc1[0], snippets[0].size(0))
        top5.update(acc5[0], snippets[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    main()
