from os import listdir
from os.path import join

import torch
from torch.utils.data.dataset import Dataset as tc_Dataset
from torchvision.io.image import read_image

from typing import Tuple, Dict
from collections import OrderedDict


class VAD_Dataset(tc_Dataset):
    def __init__(self, root_dir: str, snippet_len: int, snippet_itv: int, frm_name_len: str, n_crop: int = 3):
        '''
        root_dir: the dir containing extracted frames, e.g., 'root_dir'/01_001/[000000.jpg, ...]
        snippet_len: length of a snippet
        snippet_itv: sampling rate (interval between frames)
        frm_name_len: length of the frame name, e.g., '000000.jpg': frm_name_len=6
        '''
        super().__init__()
        self.root_dir = root_dir
        self.snippet_len = snippet_len
        self.snippet_itv = snippet_itv
        self.frm_name_len = frm_name_len
        self.n_crop = n_crop

        self.vid_len_dict = self._get_video_length()
        self.vid_name_list = sorted(list(self.vid_len_dict.keys()))

        self.sta_frm_dict = self._setup_start_frm_idx()

    def _get_video_length(self) -> Dict[str, int]:
        vid_len_dict = {}
        vid_names = sorted(listdir(self.root_dir))
        for vid_name in vid_names:
            n_img = len(listdir(join(self.root_dir, vid_name)))
            assert n_img % self.n_crop == 0, vid_name
            vid_len_dict[vid_name] = n_img // self.n_crop
        return vid_len_dict

    def _setup_start_frm_idx(self) -> OrderedDict:
        vid_name_list = sorted(listdir(self.root_dir))
        vid_sta_frm_dict = OrderedDict()
        for vid_name in vid_name_list:
            vid_sta_frm_dict[vid_name] = self.vid_len_dict[vid_name] - 1 - (self.snippet_len - 1) * self.snippet_itv
        return vid_sta_frm_dict

    def sample_frms_idx(self, sta_frm_idx) -> torch.Tensor:
        frm_idx_list = [sta_frm_idx + i * self.snippet_itv for i in range(0, self.snippet_len)]
        return torch.as_tensor(frm_idx_list)

    def _sample_all_frms(self, vid_name) -> torch.Tensor:
        frm_stack = []
        for _i_crop in range(1, self.n_crop + 1):
            frm_path_list = [join(self.root_dir, vid_name, f"{str(frm_idx).zfill(self.frm_name_len)}_{_i_crop}.jpg") for frm_idx in range(self.vid_len_dict[vid_name])]
            frm_stack.append(torch.stack([read_image(frm_path) for frm_path in frm_path_list]))
        return torch.stack(frm_stack, dim=1)

    def __getitem__(self, vid_idx) -> Tuple[str, torch.Tensor]:
        vid_name = self.vid_name_list[vid_idx]
        vid_stack = self._sample_all_frms(vid_name)  # vid_stack: [vid_len, n_crop, C, H, W]

        return vid_name, vid_stack

    def __len__(self):
        return len(self.vid_name_list)
