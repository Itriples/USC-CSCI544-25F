import math
import random
from typing import Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, BatchSampler, Dataset


class ConsecutiveBatchSampler(BatchSampler):
    '''
    Given a dataset length N and a batch size,
    each output is a "non-overlapping continuous batch" of the form [0,1,2,...,B-1], [B, B+1, ..., 2B-1], ...
    If you want to "slide continuous batches", change the step size of the range to the stride.
    '''

    def __init__(self, dataset_len: int, batch_size: int, drop_last: bool = True, stride: Optional[int] = None):
        super(ConsecutiveBatchSampler, self).__init__()
        self.N = dataset_len
        self.B = batch_size
        self.drop_last = drop_last
        self.stride = batch_size if stride is None else stride

        self.last_start = dataset_len - batch_size
        if self.last_start < 0 and drop_last:
            raise ValueError("dataset too small for given batch_size")

    def __iter__(self) -> Iterator[list[int]]:
        i = 0
        while True:
            if i + self.B > self.N:
                if not self.drop_last:
                    yield list(range(i, self.N))
                break
            yield list(range(i, i + self.B))
            i += self.stride

    def __len__(self) -> int:
        if self.drop_last:
            return max(0, (self.N - self.B) // self.stride + 1)
        else:
            return math.ceil(self.N / self.stride)


class NeighborPairBatchSampler(Sampler[list]):
    """
    For each batch: randomly select B/2 anchor points i,
    and then incorporate their neighbors i + delta to form paired samples.
    This way, any random point will be in the same batch as its consecutive neighbors,
    facilitating NCL (Non-Clearing Count).
    """

    def __init__(self, dataset_len: int, batch_size: int, k: int = 1, drop_last: bool = True):
        super(NeighborPairBatchSampler, self).__init__()
        assert batch_size % 2 == 0
        self.N = dataset_len
        self.B = batch_size
        self.k = k
        self.drop_last = drop_last

    def __iter__(self):
        # Randomly shuffle all indexes that can be used as anchor points (ensuring there is a neighbor i+delta).
        anchors = list(range(0, self.N - self.k))
        random.shuffle(anchors)
        i = 0
        while i + self.B // 2 <= len(anchors):
            batch = []
            for _ in range(self.B // 2):
                a = anchors[i]
                i += 1
                b = a + self.k
                batch.extend([a, b])
            yield batch

        if not self.drop_last and i < len(anchors):
            # Remainder handling: Try to make them pair.
            remain = anchors[i:]
            batch = []
            for a in remain:
                b = min(a + self.k, self.N - 1)
                batch.extend([a, b])
                if len(batch) >= self.B:
                    break
            if batch:
                yield batch[:self.B]

    def __len__(self):
        pairs = self.N - self.k
        full = (pairs // (self.B // 2))
        if self.drop_last:
            return full
        else:
            return full + (1 if pairs % (self.B // 2) else 0)


class TimeDataAugment(nn.Module):
    def __init__(self, jitter_std: float = 0.01, mask_ratio: float = 0.0):
        super(TimeDataAugment, self).__init__()
        self.jitter_std = jitter_std
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian dithering
        if self.jitter_std > 0:
            noise = torch.rand_like(x) * self.jitter_std
            x = x + noise
        # Randomized time mask (optional)
        if self.mask_ratio > 0:
            B, L, M = x.shape
            mask_L = max(1, int(L * self.mask_ratio))
            idx = torch.randint(0, L, (B, mask_L), device=x.device)
            x[torch.arange(B).unsqueeze(1), idx, :] = 0.0
        return x


class Projector(nn.Module):
    '''
    Input: (B, D) or (B, M, D) pooled to (B, D)
    Output: (B, d_proj)
    '''

    def __init__(self, in_dim: int, proj_dim: int = 256):
        super(Projector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim)
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = self.norm(x)
        x = F.normalize(x, dim=-1)
        return x


class NeighbourhoodContrastiveLoss(nn.Module):
    """
    Assume the samples within a batch are ordered chronologically.
    Samples where |i-j| <= k (j≠i) are considered positive examples of i, and the rest are negative examples.
    Supports two views: view1 and view2 (enhanced or different branch outputs).
    """

    def __init__(self, temperature: float = 0.07, k: int = 1, pool: bool = True):
        super(NeighbourhoodContrastiveLoss, self).__init__()
        self.T = temperature
        self.k = k
        self.pool = pool

    def forward(self, x: torch.Tensor, y: torch.Tensor, owner_rows: list[int] = None) -> torch.Tensor:
        """
        x, y: (B, M, L) Embeddings of two views (L2 normalized)
        Return: scalar InfoNCE loss
        """
        assert x.dim() == 3 and y.dim() == 3
        B1, M1, L1 = x.shape
        B2, M2, L2 = y.shape
        assert M1 == M2 and L1 == L2

        device = x.device

        # Construct the "owner" mask: owner_mask[i, j]=True indicates that y[j] is a positive instance of x[i]
        owner_mask = torch.zeros((B1, B2), dtype=torch.bool, device=device)
        for col, row in enumerate(owner_rows):
            if row < B1:
                owner_mask[row, col] = True

        if self.pool:
            x = F.normalize(x.mean(dim=1), dim=-1)
            y = F.normalize(y.mean(dim=1), dim=-1)

            pos_mask = owner_mask
        else:
            # Token level: No pooling
            # Shape flattened into (B*M, L) and (C*M, L)
            x = F.normalize(x.reshape(B1 * M1, L1), dim=-1)  # (BM, L)
            y = F.normalize(y.reshape(B2 * M2, L2), dim=-1)  # (CM, L)

            # Construct pos_mask for token-level positive examples: (BM, CM)
            # Rule: Same anchor + same token id is a positive example
            # First extend the owner_mask of (B,C) to the token level: Kronecker product (⊗) over the identity matrix I_M
            eye_M = torch.eye(M1, device=device, dtype=torch.bool)  # (M, M)
            pos_mask = torch.kron(owner_mask, eye_M)  # (B1*B2)⊗(M*M) = (BM, CM)

        # (B1, B2) Similarity
        logits = torch.matmul(x, y.t()) / self.T

        # Numerical stability: Subtract the max value of each row
        logits = logits - logits.max(dim=1, keepdim=True).values

        # For each i, take all positive examples j as the numerator (multiple positive examples are allowed)
        # The denominator is all (positive + negative) samples
        # Equivalent to InfoNCE with multiple positive examples:
        exp_logits = torch.exp(logits)
        pos_sum = (exp_logits * pos_mask.float()).sum(dim=1) + 1e-12
        total_sum = exp_logits.sum(dim=1) + 1e-12
        loss = -torch.log(pos_sum / total_sum).mean()
        return loss


class NCL(nn.Module):
    '''
    The intermediate features of the CALF temporal branch are processed through pooling, projection, and NCL loss.
    '''

    def __init__(self, d_model: int, proj_dim: int, temperature: float = 0.07, k: int = 1, pool: bool = True):
        super(NCL, self).__init__()
        self.projector = Projector(d_model, proj_dim)
        self.ncl = NeighbourhoodContrastiveLoss(temperature, k, pool)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        x, y : (B, M, d_model)`
        Here, we perform average pooling on the M channels to obtain the sequence-level embedding (B, d_model).
        Alternatively, we can skip pooling and use channel-level NCL (treating B*M as samples),
        but then we need to simultaneously construct the neighborhood relationships of (B*M, B*M).
        '''

        # x = self.projector(x)
        # y = self.projector(y)

        return self.ncl(x, y)
