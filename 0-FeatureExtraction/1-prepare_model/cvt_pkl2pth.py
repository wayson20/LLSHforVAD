from os.path import join, dirname, basename, exists
from os import removedirs
import torch
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.models import build_model
import slowfast.utils.checkpoint as cu


def cvt_model(cfg):
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)

    model = model.cpu()

    dst_dir = dirname(cfg.TEST.CHECKPOINT_FILE_PATH)
    dst_name = basename(cfg.TEST.CHECKPOINT_FILE_PATH).split('.')[0] + '.pth'
    torch.save(model, join(dst_dir, dst_name))
    return dst_dir, dst_name


if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    launch_job(cfg=cfg, init_method=args.init_method, func=cvt_model)

    dst_dir, dst_name = cvt_model(cfg)
    print(f"The '{dst_name}' file has been saved in '{dst_dir}'")

    empty_dir = './checkpoints'
    if exists(empty_dir):
        removedirs(empty_dir)
