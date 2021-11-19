import os
from os.path import join, exists, dirname
import numpy as np

import torch
from torch import multiprocessing as mp
from torch.utils.data._utils.collate import default_collate

from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.datasets.utils import tensor_normalize
from slowfast.datasets import transform

from dataset import VAD_Dataset
from network import rebuild_slowfast, pack_pathway_output


def extract_feature_sp(proc_id, num_gpus, num_procs, cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)

    gpu_id = proc_id % num_gpus

    dataset = VAD_Dataset(root_dir=cfg.DATA.PATH_TO_DATA_DIR, snippet_len=cfg.DATA.NUM_FRAMES, snippet_itv=cfg.DATA.SAMPLING_RATE, frm_name_len=len(cfg.DATA.PATH_PREFIX))

    model: torch.nn.Module = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH, map_location=f'cuda:{gpu_id}')
    model = rebuild_slowfast(model)
    model = model.cuda(gpu_id)

    # eval() mode
    model.eval()

    # spatial sampling settings
    _scale = cfg.DATA.TEST_CROP_SIZE
    max_scale = _scale
    min_scale = _scale
    crop_size = _scale

    _dmean = torch.as_tensor(cfg.DATA.MEAN)
    _dstd = torch.as_tensor(cfg.DATA.STD)

    with torch.no_grad():
        for i_vid in range(proc_id, len(dataset), num_procs):
            vid_name, frame_stack = dataset[i_vid]

            print(f"proc {proc_id} ({i_vid+1}/{len(dataset)}): {vid_name}")

            _save_path = join(cfg.TEST.SAVE_RESULTS_PATH, f"{vid_name}.pth")
            if not exists(dirname(_save_path)):
                os.makedirs(dirname(_save_path))

            if exists(_save_path):
                print(f"{_save_path} exists. Skip.")

            feature_dict = {}

            frame_stack: torch.Tensor  # NCHW

            frame_stack = frame_stack.permute(0, 2, 3, 1)  # -> NHWC
            frame_stack = tensor_normalize(frame_stack, _dmean, _dstd)
            frame_stack = frame_stack.permute(0, 3, 1, 2)  # -> NCHW
            frame_stack, _ = transform.random_short_side_scale_jitter(frame_stack, min_scale, max_scale)

            frame_stack = frame_stack.cuda(gpu_id)

            # temporal sampling
            for sta_idx in range(dataset.sta_frm_dict[vid_name] + 1):
                frms_idx = dataset.sample_frms_idx(sta_idx)
                frm_clip = frame_stack[frms_idx]  # TCHW

                inputs = []
                # spatial sampling
                for _s_idx in range(3):
                    frames, _ = transform.uniform_crop(frm_clip, crop_size, _s_idx)
                    frames = frames.permute(1, 0, 2, 3)  # -> CTHW
                    frames = pack_pathway_output(cfg, frames)
                    inputs.append(frames)
                inputs = default_collate(inputs)

                outputs = model(inputs).detach().cpu() # [3, 1, 2, 2, 2304]
                feature_dict[sta_idx] = outputs

            torch.save(feature_dict, _save_path)


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    empty_dir = './checkpoints'
    if exists(empty_dir):
        os.removedirs(empty_dir)

    num_gpus = torch.cuda.device_count()
    num_procs = cfg.DATA_LOADER.NUM_WORKERS
    mp.spawn(extract_feature_sp, (num_gpus, num_procs, cfg), nprocs=num_procs)
