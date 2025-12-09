import math
import random
from typing import Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler, BatchSampler


class ConsecutiveBatchSampler(BatchSampler):
    """
    Given a dataset length N and a batch size B,
    each output is a non-overlapping continuous batch:
    [0,1,...,B-1], [B,...,2B-1], ...
    """

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
    and then include their neighbors i + k to form pairs.
    """

    def __init__(self, dataset_len: int, batch_size: int, k: int = 1, drop_last: bool = True):
        super(NeighborPairBatchSampler, self).__init__()
        assert batch_size % 2 == 0
        self.N = dataset_len
        self.B = batch_size
        self.k = k
        self.drop_last = drop_last

    def __iter__(self):
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
    """
    Simple time-series augmentation: Gaussian jitter + optional random masking.
    x is expected to be (B, S, D) (S can be time or channel dimension depending on usage).
    """

    def __init__(self, jitter_std: float = 0.01, mask_ratio: float = 0.0):
        super(TimeDataAugment, self).__init__()
        self.jitter_std = jitter_std
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise
        if self.jitter_std > 0:
            noise = torch.randn_like(x) * self.jitter_std
            x = x + noise
        # Random mask (set some positions to 0)
        if self.mask_ratio > 0:
            B, S, D = x.shape
            mask_S = max(1, int(S * self.mask_ratio))
            idx = torch.randint(0, S, (B, mask_S), device=x.device)
            x[torch.arange(B).unsqueeze(1), idx, :] = 0.0
        return x


class Projector(nn.Module):
    """
    Input: (B, D)
    Output: (B, proj_dim) normalized
    """

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
    NCL: samples within a batch are two "views".
    x, y: (B, M, L)
    owner_rows optionally defines which rows are positives.
    """

    def __init__(self, temperature: float = 0.07, k: int = 1, pool: bool = True):
        super(NeighbourhoodContrastiveLoss, self).__init__()
        self.T = temperature
        self.k = k
        self.pool = pool

    def forward(self, x: torch.Tensor, y: torch.Tensor, owner_rows: Optional[list[int]] = None) -> torch.Tensor:
        """
        x, y: (B, M, L)
        owner_rows: optional list of length B2, owner_rows[j] = i means y[j] is positive for x[i].
                    If None, we use identity (i ↔ i).
        """
        assert x.dim() == 3 and y.dim() == 3
        B1, M1, L1 = x.shape
        B2, M2, L2 = y.shape
        assert M1 == M2 and L1 == L2

        device = x.device

        # Build owner mask
        if owner_rows is None:
            owner_mask = torch.eye(B1, B2, dtype=torch.bool, device=device)
        else:
            assert len(owner_rows) == B2, "owner_rows length must equal batch size of y (B2)."
            owner_mask = torch.zeros((B1, B2), dtype=torch.bool, device=device)
            for col, row in enumerate(owner_rows):
                if 0 <= row < B1:
                    owner_mask[row, col] = True

        if self.pool:
            # Pool over M (channels)
            x = F.normalize(x.mean(dim=1), dim=-1)  # (B1, L1)
            y = F.normalize(y.mean(dim=1), dim=-1)  # (B2, L2)
            pos_mask = owner_mask
        else:
            # Token-level
            x = F.normalize(x.reshape(B1 * M1, L1), dim=-1)  # (B1*M1, L1)
            y = F.normalize(y.reshape(B2 * M2, L2), dim=-1)  # (B2*M2, L2)
            eye_M = torch.eye(M1, device=device, dtype=torch.bool)
            pos_mask = torch.kron(owner_mask, eye_M)         # (B1*M1, B2*M2)

        logits = torch.matmul(x, y.t()) / self.T
        logits = logits - logits.max(dim=1, keepdim=True).values  # numerical stability

        exp_logits = torch.exp(logits)
        pos_sum = (exp_logits * pos_mask.float()).sum(dim=1) + 1e-12
        total_sum = exp_logits.sum(dim=1) + 1e-12

        loss = -torch.log(pos_sum / total_sum).mean()
        return loss


class NCL(nn.Module):
    """
    NCL head for CALF: given two views of temporal features (B, M, d_model),
    project and apply NeighbourhoodContrastiveLoss.
    """

    def __init__(self, d_model: int, proj_dim: int = 256, temperature: float = 0.07, k: int = 1, pool: bool = True):
        super(NCL, self).__init__()
        self.projector = Projector(d_model, proj_dim)
        self.ncl = NeighbourhoodContrastiveLoss(temperature, k, pool)

    def forward(self, x: torch.Tensor, y: torch.Tensor, owner_rows: Optional[list[int]] = None) -> torch.Tensor:
        """
        x, y: (B, M, d_model)
        """
        # Pool over M (channels) then project
        x = self.projector(x.mean(dim=1))  # (B, proj_dim)
        y = self.projector(y.mean(dim=1))  # (B, proj_dim)

        # Lift back to (B, 1, proj_dim)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        return self.ncl(x, y, owner_rows)

