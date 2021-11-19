from os.path import join, dirname, exists
import random

import torch
from torch.utils.data.dataset import Dataset as tc_Dataset
from typing import Tuple


class TrainingSet(tc_Dataset):
    def __init__(self, root_dir: str, t_rand_range: int, iterations: int):
        '''
        root_dir: the dir containing training data (snippet-level-packaged files)
        t_rand_range: a nearby randomly sampling range
        iterations: to simulate how many epochs
        '''
        super().__init__()
        self.root_dir = root_dir
        self.rand_t = t_rand_range

        _info_file = f'{dirname(root_dir)}/feat_infos/feat_info_train.pth'
        if not exists(_info_file):
            raise FileNotFoundError(f"Please generate {_info_file}")
        self.feat_container: dict = torch.load(_info_file)

        self.idx2name_map = list(self.feat_container.keys())

        self.__iterations = iterations
        # The SlowFast Network outputs 3 crops for each snippet.
        self.__n_crops = 3

    def _load_snippet(self, vid_name, t_idx, s_idx):
        '''
        Load a snippet according to the given video name and index.
        '''
        pth_path = join(self.root_dir, vid_name, f'{str(t_idx).zfill(4)}.pth')
        return torch.load(pth_path)[s_idx]

    def _sample_temporal_spatial_idx(self, vid_name) -> Tuple[int, int]:
        '''
        Choose a snippet index from a video.
        '''
        t_idx = random.choice(self.feat_container[vid_name])
        s_idx = random.choice(range(self.__n_crops))
        return (t_idx, s_idx)

    def _two_transform(self, _vid_name: str, t_idx: int, s_idx: int) -> torch.Tensor:
        """
        Choose a temporally nearby snippet in the video.
        """
        _t_idx = random.choice(range(max(t_idx - self.rand_t, 0), min(t_idx + self.rand_t, len(self.feat_container[_vid_name]) - 1)))
        _s_idx = random.choices(range(self.__n_crops), weights=[0.1, 0.4, 0.4], k=1)[0]

        return self._load_snippet(_vid_name, _t_idx, _s_idx)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Sample two snippets. One is the `idx`-th snippet and another one is its nearby snippet.
        '''
        vid_idx = idx % len(self.idx2name_map)

        _vid_name = self.idx2name_map[vid_idx]
        _t_idx, _s_idx = self._sample_temporal_spatial_idx(_vid_name)

        _snippet_0: torch.Tensor = self._load_snippet(_vid_name, _t_idx, _s_idx)
        _snippet_1 = self._two_transform(_vid_name, _t_idx, _s_idx)

        return [_snippet_0, _snippet_1]

    def __len__(self):
        '''
        The length of this dataset is also determined by 'iterations'.
        '''
        return len(self.feat_container) * self.__iterations


class TestingSet(tc_Dataset):
    def __init__(self, root_dir: str):
        '''
        root_dir: the dir containing testing data (video-level-packaged files)
        '''
        super().__init__()
        self.root_dir = root_dir

        _info_file = f'{dirname(root_dir)}/feat_infos/feat_info_test.pth'
        if not exists(_info_file):
            raise FileNotFoundError(f"Please generate {_info_file}")
        self.feat_container: dict = torch.load(_info_file)

        self.idx2name_map = list(self.feat_container.keys())

    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]:
        '''
        Return a full video.
        '''
        _vid_name = self.idx2name_map[idx]
        _snippet: torch.Tensor = torch.load(join(self.root_dir, _vid_name + '.pth'))
        return _vid_name, _snippet

    def __len__(self):
        return len(self.feat_container)
