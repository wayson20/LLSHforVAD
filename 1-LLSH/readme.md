## Codes of LSH|LLSH and light-LSH|LLSH

1.  Training\
    Run `train.sh` for training LLSH. Modify this script according to your circumstances.\
    Two checkpoints will be saved in `../save.ckpts/1-LLSH/train_mmdd-HHMMSS/checkpoint_0|1.pth.tar` (0 for LSH, 1 for LLSH).\
    Training logs will be saved in `../save.logs/1-LLSH/train_mmdd-HHMMSS.log`.\
    You could create a soft link for `save.ckpts/` to save the checkpoints in another large-capacity hard disk.

2.  Testing\
    Run `test.sh` for testing LSH|LLSH. Append `--light` option to use light-LSH|LLSH.\
    'checkpoint_0.pth.tar' is for LSH and 'checkpoint_1.pth.tar' is for LLSH. Note that the `--moco_k` option should be consistent with the checkpoint.\
    Testing logs will be saved in `../save.logs/1-LLSH/test_mmdd-HHMMSS.log`.\
    Scores will be saved in `../save.results/1-LLSH/score_dict_mmdd-HHMMSS_0|1.npz`.\
    LSH|LLSH models will be saved in `../save.results/1-LLSH/llsh_inst_mmdd-HHMMSS_0|1.pth`.

**For more details of training and testing settings, please refer to the scripts.**