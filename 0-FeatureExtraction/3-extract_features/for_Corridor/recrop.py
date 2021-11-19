import argparse
from os.path import join, exists
from os import listdir, makedirs
import cv2
import numpy as np
import multiprocessing as mp


parser = argparse.ArgumentParser("Crop 3 images for Corridor dataset.")
parser.add_argument("--video_dir", type=str, required=True,
                    help="the dir containing extracted frames, e.g., 'video_dir'/209/[000000.jpg, ...] ")
parser.add_argument("--crop_dir", type=str, required=True,
                    help="the dir to save cropped images, e.g. 'crop_dir'/(209/[000000_1.jpg, 000000_2.jpg, 000000_3.jpg ...])")
parser.add_argument('--workers', type=int, default=48,
                    help="The number of processes.")

args = parser.parse_args()
vid_root = args.video_dir
dst_root = args.crop_dir
workers: int = args.workers


def recrop(vid_name):
    print(vid_name)

    vid_dir = join(vid_root, vid_name)
    dst_dir = join(dst_root, vid_name)
    if not exists(dst_dir):
        makedirs(dst_dir)

    for frm_name in sorted(listdir(vid_dir)):
        src_img_path = join(vid_dir, frm_name)
        assert exists(src_img_path), f"'{src_img_path}' does not exits!"

        src_img: np.ndarray = cv2.imread(src_img_path)
        img_id, img_suf = frm_name.split('.')

        _crop_h = 896
        _crop_w = _crop_h
        _lfh, _lfw = 1080 - _crop_h, 256 + 64
        _dst_h = 256
        _dst_w = 256
        img1 = src_img[_lfh:_lfh + _crop_h, _lfw:_lfw + _crop_w]
        img1 = cv2.resize(img1, (_dst_w, _dst_h))

        _crop_h = 412
        _crop_w = _crop_h
        _lfh, _lfw = 64 + 32, 512 + 64
        _dst_h = 256
        _dst_w = 256
        img2 = src_img[_lfh:_lfh + _crop_h, _lfw:_lfw + _crop_w]
        img2 = cv2.resize(img2, (_dst_w, _dst_h))

        _crop_h = 256
        _crop_w = _crop_h
        _lfh, _lfw = 0, 512 + 128 + 32
        _dst_h = 256
        _dst_w = 256
        img3 = src_img[_lfh:_lfh + _crop_h, _lfw:_lfw + _crop_w]
        img3 = cv2.resize(img3, (_dst_w, _dst_h))

        cv2.imwrite(join(dst_dir, f"{img_id}_1.{img_suf}"), img1)
        cv2.imwrite(join(dst_dir, f"{img_id}_2.{img_suf}"), img2)
        cv2.imwrite(join(dst_dir, f"{img_id}_3.{img_suf}"), img3)


if __name__ == "__main__":
    pool = mp.Pool(workers)
    for vid_name in sorted(listdir(vid_root)):
        pool.apply_async(recrop, args=(vid_name,))
    pool.close()
    pool.join()
