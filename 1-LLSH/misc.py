import os
from os.path import join, exists, dirname, basename, abspath, realpath
import argparse
import time
import logging
import shutil
import torch


def get_proj_root() -> str:
    proj_root = dirname(dirname(realpath(__file__)))
    return proj_root


def get_time_stamp() -> str:
    _t = time.localtime()
    time_stamp = f"{str(_t.tm_mon).zfill(2)}{str(_t.tm_mday).zfill(2)}" + \
        f"-{str(_t.tm_hour).zfill(2)}{str(_t.tm_min).zfill(2)}{str(_t.tm_sec).zfill(2)}"
    return time_stamp


def get_logger(time_stamp, file_name: str = '', log_root='save.logs') -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if file_name:
        _save_subdir = basename(dirname(abspath(__file__)))
        log_file = join(get_proj_root(), log_root, _save_subdir, f"{basename(file_name).split('.')[0]}_{time_stamp}.log")

        if not exists(dirname(log_file)):
            os.makedirs(dirname(log_file))

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_ckpt_dir(time_stamp, file_name, ckpt_root='save.ckpts') -> str:
    _save_subdir = basename(dirname(realpath(__file__)))
    ckpt_dir = join(get_proj_root(), ckpt_root, _save_subdir, f"{basename(file_name).split('.')[0]}_{time_stamp}")
    if not exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return ckpt_dir


def get_result_dir(result_root='save.results') -> str:
    result_dir = join(get_proj_root(), result_root, basename(dirname(abspath(__file__))))
    if not exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def format_args(args: argparse.Namespace, sorted_key: bool = True) -> str:
    _cont = '\n' + '-' * 30 + "args" + '-' * 30 + '\n'
    args: dict = args.__dict__

    m_l = max([len(k) for k in args.keys()])

    key_list = list(args.keys())
    if sorted_key:
        key_list.sort()

    for _k in key_list:
        _v = args[_k]
        _cont += f"{_k:>{m_l}s} = {_v}\n"
    _cont += '-' * 60 + '\n'
    return _cont


def save_checkpoint(state, is_best, filedir, epoch) -> None:
    if not exists(filedir):
        os.makedirs(filedir)

    filename = join(filedir, f'checkpoint_{epoch}.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(filedir, 'checkpoint_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
