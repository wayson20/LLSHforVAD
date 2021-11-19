import argparse
from os import listdir, makedirs, mkdir
from os.path import join, exists
import torch
import tqdm

parser = argparse.ArgumentParser("Convert video-level-packaged feature files to snippet-level-packaged feature files.")
parser.add_argument("--src_dir", type=str, required=True,
                    help="The dir of extracted video-level-packaged features, e.g. 'src_dir'/[01_001.pth, 01_002.pth, ...]")
parser.add_argument("--dst_dir", type=str, required=True,
                    help="The dir to save snippet-level-packaged features, e.g. 'dst_dir'/(01_001/[000000.pth, 000001.pth, ...])")

args = parser.parse_args()

src_root: str = args.src_dir
dst_root: str = args.dst_dir

assert exists(src_root), f"{src_root} does not exist!"

if not exists(dst_root):
    makedirs(dst_root)

for src_name in tqdm.tqdm(sorted(listdir(src_root))):
    feat_src: dict = torch.load(join(src_root, src_name))

    dst_dir = join(dst_root, src_name.split('.')[0])
    if not exists(dst_dir):
        mkdir(dst_dir)

    for _i_frm, _feat in feat_src.items():
        torch.save(_feat, join(dst_dir, str(_i_frm).zfill(4) + '.pth'))
