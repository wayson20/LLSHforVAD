import os
from os.path import join, dirname, exists
import tqdm

import torch
from torch.utils.data.dataset import Dataset as tc_Dataset
from typing import Tuple


class TrainingSet(tc_Dataset):
    def __init__(self, root_dir: str):
        '''
        root_dir: the dir containing training data (snippet-level files)
        '''
        super().__init__()
        self.root_dir = root_dir

    def get_training_mat(self) -> torch.Tensor:
        '''
        Concatenate all training vectors as a matrix.
        '''
        train_mat = []
        for _vid_name in tqdm.tqdm(sorted(os.listdir(self.root_dir)), desc="Reading training mat"):
            vid_dir = join(self.root_dir, _vid_name)
            _snippet_mat = []

            for _snippet_name in sorted(os.listdir(vid_dir)):
                snippet_pth: torch.Tensor = torch.load(join(vid_dir, _snippet_name))
                _snippet_mat.append(snippet_pth.flatten(1))

            _snippet_mat = torch.cat(_snippet_mat, 0)
            train_mat.append(_snippet_mat)

        train_mat = torch.cat(train_mat, 0)
        return train_mat

    def __len__(self):
        '''
        The length of this dataset is also determined by 'iterations'.
        '''
        return len(self.feat_container) * self.__iterations


class TestingSet(tc_Dataset):
    def __init__(self, root_dir: str):
        '''
        root_dir: the dir containing testing data (video-level files)
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
