# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import numpy as np
import copy


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, TT=0.04, ST=0.1, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.TT = TT
        self.ST = ST

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_w", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_w = nn.functional.normalize(self.queue_w, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_w):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        keys_w = concat_all_gather(keys_w)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_w[:, ptr:ptr + batch_size] = keys_w.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    @torch.no_grad()
    def _img_interpolation(self, im_q, im_k):
        batchSize = im_q.size(0)
        noise = torch.randperm(batchSize)
        # 0-mixup  1-cutmix
        choose = 1
        # choose = np.random.randint(2)
        ratio = np.random.beta(1.0, 1.0)  #####1

        if choose == 0:  # mixup
            im_mix = ratio * im_q + (1 - ratio) * im_k[noise]
        else:  # cutmix
            bbx11, bby11, bbx12, bby12 = self._rand_bbox(im_q.size(), ratio)
            im_mix = copy.deepcopy(im_q)
            im_mix[:, :, bbx11:bbx12, bby11:bby12] = im_k[noise, :, bbx11:bbx12, bby11:bby12]
            ratio = 1 - ((bbx12 - bbx11) * (bby12 - bby11) / (im_q.size(2) * im_q.size(3)))

        return im_mix, noise, ratio

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, im_k_w):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        im_mix, noise, ratio = self._img_interpolation(im_q, im_k)

        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        q_mix = self.encoder_q(im_mix)
        q_mix = nn.functional.normalize(q_mix, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            im_k_w, idx_unshuffle_w = self._batch_shuffle_ddp(im_k_w)

            k = self.encoder_k(im_k)  # keys: NxC
            k_w = self.encoder_k(im_k_w)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            k_w = nn.functional.normalize(k_w, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_w = self._batch_unshuffle_ddp(k_w, idx_unshuffle_w)

        reordered_k = k[noise]
        f_mix = ratio * q + (1 - ratio) * reordered_k
        f_mix = nn.functional.normalize(f_mix, dim=1)
        f_mix = f_mix.detach()

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # compute distributions
        logits_q = torch.einsum('nc,ck->nk', [q, self.queue_w.clone().detach()])
        logits_k = torch.einsum('nc,ck->nk', [k_w, self.queue_w.clone().detach()])

        logits_q /= self.ST
        logits_k /= self.TT

        ## local
        l_pos_local = torch.einsum('nc,nc->n', [q_mix, f_mix.detach()]).unsqueeze(-1)
        l_neg_local = torch.einsum('nc,ck->nk', [q_mix, self.queue.clone().detach()])
        logits_local = torch.cat([l_pos_local, l_neg_local], dim=1)
        logits_local /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, k_w)

        return logits, logits_local, labels, logits_q, logits_k.detach()


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
