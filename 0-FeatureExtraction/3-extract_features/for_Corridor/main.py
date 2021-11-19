import os
from os.path import join, exists, dirname
import numpy as np

import torch
from torch import multiprocessing as mp
from torch.utils.data._utils.collate import default_collate

from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.datasets.utils import tensor_normalize

from dataset import VAD_Dataset
from network import rebuild_slowfast, pack_pathway_output


def extract_feature_long_sp(proc_id, num_gpus, num_procs, cfg):
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed(cfg.RNG_SEED)

    gpu_id = proc_id % num_gpus

    n_crop = 3  # P=3
    dataset = VAD_Dataset(root_dir=cfg.DATA.PATH_TO_DATA_DIR, snippet_len=cfg.DATA.NUM_FRAMES, snippet_itv=cfg.DATA.SAMPLING_RATE,
                          frm_name_len=len(cfg.DATA.PATH_PREFIX), n_crop=n_crop)

    model: torch.nn.Module = torch.load(cfg.TEST.CHECKPOINT_FILE_PATH, map_location=f'cuda:{gpu_id}')
    model = rebuild_slowfast(model)
    model = model.cuda(gpu_id)

    model.eval()

    _dmean = torch.as_tensor(cfg.DATA.MEAN)
    _dstd = torch.as_tensor(cfg.DATA.STD)

    with torch.no_grad():
        for i_vid in range(proc_id, len(dataset), num_procs):
            vid_name = dataset.vid_name_list[i_vid]

            _save_path = join(cfg.TEST.SAVE_RESULTS_PATH, f"{vid_name}.pth")

            if exists(_save_path):
                print(f"{_save_path} exists. Skip.")

            if not exists(dirname(_save_path)):
                os.makedirs(dirname(_save_path))

            print(f"proc {proc_id} ({i_vid+1}/{len(dataset)}): {vid_name}")
            vid_name, frame_stack = dataset[i_vid]

            feature_dict = {}
            # --------------------- transforms ---------------------
            frame_stack: torch.Tensor  # NPCHW
            n_frm = frame_stack.shape[0]
            frame_stack = frame_stack.reshape([-1, 3, 256, 256])  # NP,C,H,W
            frame_stack = frame_stack.permute(0, 2, 3, 1)  # -> NP,H,W,C

            try:
                frame_stack = tensor_normalize(frame_stack, _dmean, _dstd)
            except RuntimeError as e:
                print("+" * 90)
                print(f"{proc_id}: {i_vid}, \"{vid_name}\" ERROR!")
                print(f"{e}")
                print(f"*** It may be caused by limited memory. ***")
                print("=" * 90)
                continue

            frame_stack = frame_stack.permute(0, 3, 1, 2)  # -> NP,C,H,W
            # ------------------------------------------------------

            # temporal sampling
            frame_stack = frame_stack.reshape([n_frm, n_crop, 3, 256, 256])  # N,P,C,H,W

            for sta_idx in range(dataset.sta_frm_dict[vid_name] + 1):
                frms_idx = dataset.sample_frms_idx(sta_idx)
                frames = frame_stack[frms_idx]  # T,P,C,H,W

                frames = frames.cuda(gpu_id)

                frames = frames.permute(1, 2, 0, 3, 4)  # -> P,C,T,H,W
                inputs = default_collate([pack_pathway_output(cfg, _frm) for _frm in frames])

                outputs = model(inputs).detach().cpu()  # [3, 1, 2, 2, 2304]
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

    mp.spawn(extract_feature_long_sp, (num_gpus, num_procs, cfg), nprocs=num_procs)
