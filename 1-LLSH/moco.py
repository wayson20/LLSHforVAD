import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed

'''
Reference:
https://github.com/facebookresearch/moco
'''


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    """

    def __init__(self, moco_K: int, moco_m: float, moco_T: float, queue_dim: int, Network, **net_args):

        super(MoCo, self).__init__()

        self.K = moco_K
        self.m = moco_m
        self.T = moco_T

        self.encoder_q = Network(**net_args)
        self.encoder_k = Network(**net_args)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(queue_dim, moco_K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        '''
        Gather keys before updating queue
        '''
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        idx_shuffle = torch.randperm(batch_size_all).cuda()

        torch.distributed.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)

        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, data_q, data_k):
        z_q = self.encoder_q(data_q)  # z_q: NxC
        z_q = F.normalize(z_q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()

            im_k, idx_unshuffle = self._batch_shuffle_ddp(data_k)

            z_k = self.encoder_k(im_k)  # z_k: NxC
            z_k = F.normalize(z_k, dim=1)

            z_k = self._batch_unshuffle_ddp(z_k, idx_unshuffle)

        l_pos = torch.einsum('nc,nc->n', [z_q, z_k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [z_q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        self._dequeue_and_enqueue(z_k)

        return logits, labels


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
