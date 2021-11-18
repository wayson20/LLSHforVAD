import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score


def cal_macro_auc(score_dict: dict, gt_npz: np.lib.npyio.NpzFile, K: int):
    '''
    Calculate macro-AUC.
    '''
    auc_dict = {}

    vid_name_list = sorted(list(score_dict.keys()))
    for vid_name in vid_name_list:
        vid_gts = gt_npz[vid_name]

        val_score: np.ndarray = score_dict[vid_name][K]
        val_score = gaussian_filter1d(val_score, sigma=12)

        vid_scores: np.ndarray = np.zeros_like(vid_gts, dtype=val_score.dtype)

        assert len(val_score.shape) == len(vid_scores.shape) == 1

        dif_len = len(vid_scores) - len(val_score)
        offset = dif_len // 2 + 1

        vid_scores[offset:offset + len(val_score)] = val_score[:]

        assert vid_gts.shape == vid_scores.shape, f"{vid_gts.shape}, {vid_scores.shape}"

        if np.all(vid_gts == vid_gts[0]):
            vid_gts[0] = 1 - vid_gts[0]

        vid_auc = roc_auc_score(vid_gts, vid_scores)

        auc_dict[vid_name] = vid_auc

    macro_auc = np.mean(np.array(list(auc_dict.values())))
    return macro_auc


def cal_micro_auc(score_dict: dict, gt_npz: np.lib.npyio.NpzFile, K):
    '''
    Calculate micro-AUC.
    '''
    cat_gts = []
    cat_scores = []

    vid_name_list = sorted(list(score_dict.keys()))

    for vid_name in vid_name_list:
        vid_gts = gt_npz[vid_name]
        val_score: np.ndarray = score_dict[vid_name][K]
        val_score = gaussian_filter1d(val_score, sigma=12)

        if val_score.max() != val_score.min():
            val_score = (val_score - val_score.min()) / (val_score.max() - val_score.min())
        else:
            val_score: np.ndarray = score_dict[vid_name]

        vid_scores: np.ndarray = np.zeros_like(vid_gts, dtype=val_score.dtype)
        assert len(val_score.shape) == len(vid_scores.shape) == 1

        dif_len = len(vid_scores) - len(val_score)
        offset = dif_len // 2 + 1
        vid_scores[offset:offset + len(val_score)] = val_score[:]

        assert vid_gts.shape == vid_scores.shape, f"{vid_gts.shape}, {vid_scores.shape}"

        cat_gts.append(vid_gts)
        cat_scores.append(vid_scores)

    cat_scores = np.concatenate(cat_scores)
    cat_gts = np.concatenate(cat_gts)

    micro_auc = roc_auc_score(cat_gts, cat_scores)

    return micro_auc
