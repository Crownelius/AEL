"""Quaternion-output placement head: maps synsets to S^3 ⊂ R^4.

Network: Embedding(vocab) -> Linear -> normalize. Compact codomain (the
3-sphere) means no acosh instability, no t-blowup, no rescaling tricks.

Loss: contrastive in S^2 distance after Hopf projection. Negatives are
pushed apart on S^2; positives pulled together.

Snap to gasket: only the base point (S^2) is snapped. The phase stays
continuous, which preserves the residual signal that pure gasket-index
snapping loses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hopf_map_t(q: torch.Tensor) -> torch.Tensor:
    """Differentiable Hopf map S^3 -> S^2. q: (..., 4) unit, returns (..., 3)."""
    a, b, c, d = q.unbind(dim=-1)
    x = 2.0 * (a * c + b * d)
    y = 2.0 * (b * c - a * d)
    z = a * a + b * b - c * c - d * d
    return torch.stack([x, y, z], dim=-1)


def sphere_dist_t(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Differentiable spherical distance on S^2.

    Uses the arccos formulation with the standard stable bound on the
    dot-product. Unlike acosh, arccos has bounded gradient except at the
    endpoints, so a single epsilon clamp suffices.
    """
    ip = (p * q).sum(dim=-1)
    ip = ip.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(ip)


def chord_dist2_t(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Chord-squared distance ||p - q||^2 on S^2. Smooth at antipodes, no
    arccos. Monotone in the spherical distance: 2(1 - cos d)."""
    return ((p - q) ** 2).sum(dim=-1)


class HopfPlacementHead(nn.Module):
    """Synset -> quaternion in S^3. Hopf project to S^2 for base; phase
    via phase_of() at inference time."""

    def __init__(self, vocab_size: int, hidden_dim: int = 64, init_scale: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.normal_(self.embedding.weight, std=init_scale)
        self.proj = nn.Linear(hidden_dim, 4)
        nn.init.normal_(self.proj.weight, std=init_scale)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Returns unit quaternions, shape (..., 4)."""
        h = self.embedding(idx)
        q = self.proj(h)
        q = F.normalize(q, dim=-1, eps=1e-8)
        return q

    def base(self, idx: torch.Tensor) -> torch.Tensor:
        """Hopf-project to S^2."""
        return hopf_map_t(self.forward(idx))


@dataclass
class HopfTrainConfig:
    epochs: int = 100
    batch_size: int = 256
    lr: float = 5e-3
    n_negatives: int = 5
    margin: float = 0.5            # in S^2 distance (radians)
    grad_clip: float = 5.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 20
    use_chord_dist: bool = True    # chord^2 is smoother; arccos is geodesic


def train_hopf_head(
    head: HopfPlacementHead,
    pos_edges: list[tuple[int, int]],
    sib_edges: list[tuple[int, int]],
    vocab_size: int,
    cfg: HopfTrainConfig,
) -> dict[str, float]:
    device = torch.device(cfg.device)
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr)
    pos = torch.tensor(pos_edges, dtype=torch.long, device=device)
    sib = torch.tensor(sib_edges, dtype=torch.long, device=device) if sib_edges else None
    all_pos = pos if sib is None else torch.cat([pos, sib], dim=0)

    def dist(a, b):
        if cfg.use_chord_dist:
            return chord_dist2_t(a, b)
        return sphere_dist_t(a, b)

    last_loss = float("nan")
    for ep in range(cfg.epochs):
        perm = torch.randperm(all_pos.size(0), device=device)
        shuf = all_pos[perm]
        losses = []
        for start in range(0, shuf.size(0), cfg.batch_size):
            batch = shuf[start : start + cfg.batch_size]
            a = batch[:, 0]; b = batch[:, 1]
            neg = torch.randint(0, vocab_size, (a.size(0), cfg.n_negatives), device=device)

            p_a = head.base(a)
            p_b = head.base(b)
            p_neg = head.base(neg.reshape(-1)).reshape(neg.size(0), cfg.n_negatives, 3)

            d_pos = dist(p_a, p_b)
            d_neg = dist(p_a.unsqueeze(1).expand(-1, cfg.n_negatives, -1), p_neg)

            loss = F.relu(d_pos.unsqueeze(1) - d_neg + cfg.margin).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))

        if (ep + 1) % cfg.log_every == 0:
            ml = sum(losses) / len(losses)
            print(f"  epoch {ep+1:3d}/{cfg.epochs}  loss={ml:.4f}")
            last_loss = ml

    return {"final_loss": last_loss}


@torch.no_grad()
def export_quaternions(head: HopfPlacementHead, vocab_size: int) -> np.ndarray:
    device = next(head.parameters()).device
    idx = torch.arange(vocab_size, device=device)
    return head(idx).cpu().numpy()


@torch.no_grad()
def export_bases(head: HopfPlacementHead, vocab_size: int) -> np.ndarray:
    device = next(head.parameters()).device
    idx = torch.arange(vocab_size, device=device)
    return head.base(idx).cpu().numpy()


def snap_to_gasket_base(
    bases: np.ndarray, gasket_s2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For each learned base point on S^2, find the index of the nearest
    gasket-base point on S^2.

    Returns (snapped_indices, residual_quaternion_phase_proxy) where the
    residual is just sphere_dist between learned and snapped base (useful
    diagnostic for how much the snap moved each point).
    """
    # Vectorised: dot products N_learned x N_gasket.
    ip = bases @ gasket_s2.T
    ip = np.clip(ip, -1.0, 1.0)
    idx = ip.argmax(axis=1)
    snap_dist = np.arccos(ip[np.arange(len(idx)), idx])
    return idx, snap_dist
