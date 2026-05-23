"""Learnable placement: small PyTorch module mapping WordNet synsets to UHS H^3.

Module structure:
  - one learnable embedding per synset (free positions in R^d)
  - linear projection to (x, y, t_pre)
  - softplus + small epsilon ensures t > 0 (upper-half-space constraint)

Training objective: contrastive on WordNet edges.
  positives: (n, hypernym_parent), (n, sibling)
  negatives: (n, random_non_neighbor)
  loss: margin loss in UHS hyperbolic distance.

After training, each synset has a learned UHS point. We snap each point to
the nearest GASKET CIRCLE (under UHS distance) -- this lands the learned
embedding back onto the integer lattice for the retrieval system.

Twin-pair bonus: an optional reward term that pulls sibling pairs toward
gasket circles whose curvatures form a twin-prime pair.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def uhs_dist_t(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Differentiable UHS distance, numerically stable.

    Uses acosh(1 + x) = log(1 + x + sqrt(x*(x+2))). For x near 0 (small
    distances) this reduces to log(1 + sqrt(2x)) which has finite gradient,
    avoiding the singularity at acosh(1).
    """
    diff = p - q
    num = (diff ** 2).sum(dim=-1)
    pt = p[..., 2].clamp(min=1e-3)
    qt = q[..., 2].clamp(min=1e-3)
    x = num / (2.0 * pt * qt)  # = cosh(d) - 1, in [0, +inf)
    x = x.clamp(min=1e-7, max=1e7)
    # Stable acosh(1 + x).
    return torch.log1p(x + torch.sqrt(x * (x + 2.0)))


class PlacementHead(nn.Module):
    """Maps synset indices to UHS points."""

    def __init__(self, vocab_size: int, hidden_dim: int = 64, init_scale: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.embedding.weight, std=init_scale)
        self.proj = nn.Linear(hidden_dim, 3)
        nn.init.normal_(self.proj.weight, std=init_scale)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        h = self.embedding(idx)
        p = self.proj(h)
        x, y, t_pre = p.unbind(dim=-1)
        # softplus ensures t > 0; +eps to avoid 0-divides
        t = F.softplus(t_pre) + 1e-3
        return torch.stack([x, y, t], dim=-1)


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    n_negatives: int = 5
    margin: float = 0.5
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 10


def build_edge_lists(sub, synset_to_idx: dict[str, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (positive_pairs, sibling_pairs).

    positive_pairs = (child_idx, parent_idx) -- the hypernym tree.
    sibling_pairs  = (a_idx, b_idx) for siblings (children of same parent).
    """
    pos: list[tuple[int, int]] = []
    sib: list[tuple[int, int]] = []
    for name, node in sub.nodes.items():
        if name not in synset_to_idx:
            continue
        if node.parent and node.parent in synset_to_idx:
            pos.append((synset_to_idx[name], synset_to_idx[node.parent]))
    for name, node in sub.nodes.items():
        for i, a in enumerate(node.children):
            if a not in synset_to_idx:
                continue
            for b in node.children[i + 1 :]:
                if b not in synset_to_idx:
                    continue
                sib.append((synset_to_idx[a], synset_to_idx[b]))
    return pos, sib


def train_placement_head(
    head: PlacementHead,
    pos_edges: list[tuple[int, int]],
    sib_edges: list[tuple[int, int]],
    vocab_size: int,
    cfg: TrainConfig,
) -> dict[str, float]:
    """Margin loss: hyp_pairs and sib_pairs should be close; randoms far.

    Loss = sum over positives of max(0, d(pos) - d(neg) + margin)
    averaged across negatives.
    """
    device = torch.device(cfg.device)
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)

    pos = torch.tensor(pos_edges, dtype=torch.long, device=device)
    sib = torch.tensor(sib_edges, dtype=torch.long, device=device) if sib_edges else None
    all_pos = pos if sib is None else torch.cat([pos, sib], dim=0)

    last_loss = float("nan")
    for ep in range(cfg.epochs):
        perm = torch.randperm(all_pos.size(0), device=device)
        all_pos_shuf = all_pos[perm]
        losses = []
        for start in range(0, all_pos_shuf.size(0), cfg.batch_size):
            batch = all_pos_shuf[start : start + cfg.batch_size]
            a = batch[:, 0]
            b = batch[:, 1]
            # Negatives: random other synsets.
            neg = torch.randint(0, vocab_size, (a.size(0), cfg.n_negatives), device=device)

            p_a = head(a)                          # (B, 3)
            p_b = head(b)                          # (B, 3)
            p_neg = head(neg.reshape(-1)).reshape(neg.size(0), cfg.n_negatives, 3)

            d_pos = uhs_dist_t(p_a, p_b)           # (B,)
            d_neg = uhs_dist_t(p_a.unsqueeze(1), p_neg)  # (B, K)

            margin = cfg.margin
            loss = F.relu(d_pos.unsqueeze(1) - d_neg + margin).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))

        if (ep + 1) % cfg.log_every == 0:
            mean_loss = sum(losses) / len(losses)
            print(f"  epoch {ep+1:3d}/{cfg.epochs}  loss={mean_loss:.4f}")
            last_loss = mean_loss

    return {"final_loss": last_loss}


@torch.no_grad()
def export_points(head: PlacementHead, vocab_size: int) -> np.ndarray:
    device = next(head.parameters()).device
    idx = torch.arange(vocab_size, device=device)
    return head(idx).cpu().numpy()


def snap_to_nearest_circle(
    learned_points: np.ndarray,
    gasket_uhs_points: list,
    rescale: bool = True,
) -> list[int]:
    """For each learned UHS point, find the index of the closest gasket circle.

    If rescale=True, linearly map the learned (x, y, t) ranges into the gasket's
    range before snapping. This avoids the failure mode where learned t >> any
    gasket t and everything snaps to the bounding circle.
    """
    from .uhs import uhs_dist
    pts = learned_points.copy()
    if rescale:
        gx = np.array([p.x for p in gasket_uhs_points])
        gy = np.array([p.y for p in gasket_uhs_points])
        gt = np.array([p.t for p in gasket_uhs_points])

        def _rescale(vals, target_lo, target_hi):
            v_lo, v_hi = vals.min(), vals.max()
            if v_hi - v_lo < 1e-9:
                return np.full_like(vals, (target_lo + target_hi) / 2.0)
            return target_lo + (vals - v_lo) * (target_hi - target_lo) / (v_hi - v_lo)

        pts[:, 0] = _rescale(pts[:, 0], gx.min(), gx.max())
        pts[:, 1] = _rescale(pts[:, 1], gy.min(), gy.max())
        # Log-scale t since it spans orders of magnitude in the gasket.
        log_t = np.log(np.maximum(pts[:, 2], 1e-6))
        log_gt = np.log(gt)
        pts[:, 2] = np.exp(_rescale(log_t, log_gt.min(), log_gt.max()))

    out: list[int] = []
    n_circles = len(gasket_uhs_points)

    class _P:
        __slots__ = ("x", "y", "t")
        def __init__(self, xyt):
            self.x = float(xyt[0]); self.y = float(xyt[1]); self.t = max(float(xyt[2]), 1e-6)

    for p in pts:
        q = _P(p)
        best_i = 0
        best_d = float("inf")
        for i in range(n_circles):
            d = uhs_dist(q, gasket_uhs_points[i])
            if d < best_d:
                best_d = d
                best_i = i
        out.append(best_i)
    return out
