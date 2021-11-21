import argparse
from os import listdir, makedirs, mkdir
from os.path import join, exists
import torch
import tqdm

parser = argparse.ArgumentParser("Convert video-level-packaged feature files to snippet-level-packaged feature files.")
parser.add_argument("--video_level_packaged_dir", type=str, required=True,
                    help="The dir of extracted video-level-packaged features, e.g. 'src_dir'/[01_001.pth, 01_002.pth, ...]")
parser.add_argument("--snippet_level_packaged_dir", type=str, required=True,
                    help="The dir to save snippet-level-packaged features, e.g. 'dst_dir'/(01_001/[000000.pth, 000001.pth, ...])")
parser.add_argument('--snippet_pth_name_len', type=int, default=6,
                    help="length of the snippet-level-packaged files, e.g., frm_name_len=6: '000000.pth', '000001.pth', ...")

args = parser.parse_args()
src_root: str = args.video_level_packaged_dir
dst_root: str = args.snippet_level_packaged_dir
pth_name_len: int = args.snippet_pth_name_len

assert exists(src_root), f"{src_root} does not exist!"

if not exists(dst_root):
    makedirs(dst_root)

for src_name in tqdm.tqdm(sorted(listdir(src_root))):
    feat_src: dict = torch.load(join(src_root, src_name))

    dst_dir = join(dst_root, src_name.split('.')[0])
    if not exists(dst_dir):
        mkdir(dst_dir)

    for _i_frm, _feat in feat_src.items():
        torch.save(_feat, join(dst_dir, str(_i_frm).zfill(pth_name_len) + '.pth'))
