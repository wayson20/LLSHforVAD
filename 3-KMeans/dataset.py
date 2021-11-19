from os import listdir
from os.path import join
import tqdm
import torch
from torch.utils.data.dataset import Dataset as tc_Dataset
from typing import Tuple, List


class TrainingSet(tc_Dataset):
    def __init__(self, root_dir: str):
        '''
        root_dir: the dir containing training data (snippet-level-packaged files)
        '''
        super().__init__()
        self.root_dir = root_dir

    def get_training_mat(self) -> torch.Tensor:
        '''
        Concatenate all training vectors as a matrix.
        '''
        train_mat = []
        for _vid_name in tqdm.tqdm(sorted(listdir(self.root_dir)), desc="Reading training mat"):
            vid_dir = join(self.root_dir, _vid_name)
            _snippet_mat = []

            for _snippet_name in sorted(listdir(vid_dir)):
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
        return len(listdir(self.root_dir))


class TestingSet(tc_Dataset):
    def __init__(self, root_dir: str):
        '''
        root_dir: the dir containing testing data (video-level-packaged files)
        '''
        super().__init__()
        self.root_dir = root_dir

        self.feat_info = self._get_feat_info()

    def _get_feat_info(self) -> List[str]:
        '''
        Get the video names, e.g., ['01_0014', '01_0015', ...]
        '''
        feat_info = []
        for pth_name in sorted(listdir(self.root_dir)):
            vid_name = pth_name.split('.')[0]
            feat_info.append(vid_name)
        return feat_info

    def __getitem__(self, idx) -> Tuple[str, torch.Tensor]:
        '''
        Return a full video.
        '''
        _vid_name = self.feat_info[idx]
        _snippet: torch.Tensor = torch.load(join(self.root_dir, _vid_name + '.pth'))
        return _vid_name, _snippet

    def __len__(self):
        return len(self.feat_info)
