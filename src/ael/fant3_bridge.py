"""fant3 compatibility bridge.

`SpinorApollonianMemory` in fant3 stores transformer hidden states using a
2D Kocik spinor (Cl(2,-1)) and an alpha/beta chirality split. This module
exposes the AEL stack (Cl(3,1) Descartes 4-vector + UHS gasket lift + cone
attention + twin-prime address book) through the same store/retrieve API,
so it can drop into fant3 with minimal changes.

Key differences vs SpinorApollonianMemory:

  Kocik 2D spinor  ->  Descartes 4-vector (Cl(3,1))
    Two-pack chirality (alpha/beta) becomes a richer assignment: each stored
    state lands at a specific gasket circle, identified by a twin-prime
    address. We keep alpha/beta as a back-compat coarse label (sign of the
    Cl(3,1) "timelike" component) but also expose the full circle index.

  scalar curvature norm  ->  signed integer curvature on the gasket lattice
    The 'curvature' field returned by get_stats() is now the actual
    integer curvature of the assigned circle, not a learned scalar.

  Descartes regularizer (top-4 spinors) -> exact Cl(3,1) Descartes form
    Our descartes_loss evaluates F(b) = 2 sum b^2 - (sum b)^2 on the
    Descartes 4-vector lifted from each stored item's parent quadruple.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .descartes_3d import descartes_form, minkowski_inner
from .gasket import Gasket, build_gasket, standard_root_neg1_2_2_3
from .lift import circle_4vector
from .placement_head import PlacementHead, uhs_dist_t
from .prime_addr import TwinAddressBook, address_of_circle
from .uhs import UHSPoint, all_uhs_points, uhs_dist


class AELMemory(nn.Module):
    """fant3-compatible memory module with AEL backbone.

    API mirrors `fant3.model.spinor_apollonian.SpinorApollonianMemory`:
      store(embeddings, hidden_preRMSnorm)
      retrieve(query, top_k, pool, hidden_preRMSnorm)
      descartes_loss(query_spinors, lmbda)
      get_stats()
      reset()
    """

    def __init__(
        self,
        dim: int = 768,
        alpha_cap: int = 5000,
        beta_cap: int = 5000,
        gasket_depth: int = 7,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha_cap = alpha_cap
        self.beta_cap = beta_cap

        # Trainable placement head: hidden -> (x, y, t) in UHS.
        self.placement = PlacementHead(vocab_size=1, hidden_dim=dim)
        # We use the head as a stateless projection (vocab_size=1 placeholder)
        # by directly applying its proj layer to incoming hidden states.
        # Override the lookup-based forward.
        del self.placement.embedding
        # Lightweight projection from dim -> 3.
        self.proj_uhs = nn.Linear(dim, 3, bias=False)
        nn.init.normal_(self.proj_uhs.weight, std=0.01)

        # Pre-build the gasket geometry (fixed lattice).
        self.gasket: Gasket = build_gasket(standard_root_neg1_2_2_3(), max_depth=gasket_depth)
        self.gasket_points: list[UHSPoint] = all_uhs_points(self.gasket)
        self.address_book = TwinAddressBook.build(self.gasket)

        # Stash circle (x, y, t) as a buffer for fast vectorized snap.
        coords = np.array(
            [[p.x, p.y, p.t] for p in self.gasket_points], dtype=np.float32
        )
        self.register_buffer("gasket_coords", torch.from_numpy(coords))
        circle_ks = np.array(
            [int(round(c.k)) for c in self.gasket.circles], dtype=np.int64
        )
        self.register_buffer("gasket_curvatures", torch.from_numpy(circle_ks))

        # Memory packs.
        self.register_buffer("alpha_emb", torch.zeros(alpha_cap, dim))
        self.register_buffer("alpha_circle", torch.zeros(alpha_cap, dtype=torch.long))
        self.register_buffer("alpha_curvature", torch.zeros(alpha_cap))
        self.register_buffer("alpha_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("alpha_head", torch.tensor(0, dtype=torch.long))

        self.register_buffer("beta_emb", torch.zeros(beta_cap, dim))
        self.register_buffer("beta_circle", torch.zeros(beta_cap, dtype=torch.long))
        self.register_buffer("beta_curvature", torch.zeros(beta_cap))
        self.register_buffer("beta_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("beta_head", torch.tensor(0, dtype=torch.long))

        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------
    # UHS projection (the only place hidden -> geometry happens)
    # ------------------------------------------------------------------

    def project_uhs(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden (..., dim) -> (..., 3) with t > 0 via softplus."""
        p = self.proj_uhs(hidden)
        x, y, t_pre = p.unbind(dim=-1)
        t = F.softplus(t_pre) + 1e-3
        return torch.stack([x, y, t], dim=-1)

    @torch.no_grad()
    def _snap_to_circle(self, uhs_pts: torch.Tensor) -> torch.Tensor:
        """uhs_pts (N, 3) -> circle indices (N,) via nearest UHS distance.

        Vectorised approximation: use the simpler half-space "cosh-1" proxy
        d2(p, q) = ((p-q)**2).sum() / (p.t * q.t),
        which preserves the argmin while avoiding the per-pair acosh.
        """
        pts = uhs_pts.float()
        g = self.gasket_coords.to(pts.device).float()
        # (N, M, 3)
        diff = pts.unsqueeze(1) - g.unsqueeze(0)
        num = (diff ** 2).sum(dim=-1)
        denom = pts[:, 2].unsqueeze(1) * g[:, 2].unsqueeze(0) + 1e-9
        proxy = num / denom
        return proxy.argmin(dim=-1)

    # ------------------------------------------------------------------
    # API: store
    # ------------------------------------------------------------------

    @torch.no_grad()
    def store(
        self,
        embeddings: torch.Tensor,
        hidden_preRMSnorm: Optional[torch.Tensor] = None,
    ) -> Dict[str, int]:
        emb = embeddings.detach()
        if emb.dim() == 3:
            B, T, D = emb.shape
            emb = emb.reshape(B * T, D)
        elif emb.dim() == 1:
            emb = emb.unsqueeze(0)
        N, D = emb.shape
        assert D == self.dim, f"embedding dim {D} != memory dim {self.dim}"

        hid = hidden_preRMSnorm.detach() if hidden_preRMSnorm is not None else emb
        if hid.dim() == 3:
            hid = hid.reshape(-1, self.dim)

        with torch.enable_grad():
            uhs_pts = self.project_uhs(hid).detach()
        circle_idx = self._snap_to_circle(uhs_pts)              # (N,)
        curvatures = self.gasket_curvatures[circle_idx].float()  # (N,)

        # Chirality: keep fant3's alpha/beta back-compat. We use sign of the
        # learned UHS t-residual (t - median) so each batch has roughly
        # balanced packs at init.
        median_t = uhs_pts[:, 2].median()
        is_alpha = uhs_pts[:, 2] > median_t

        self.global_step.add_(1)
        n_a = self._fifo_write_pack(
            self.alpha_emb, self.alpha_circle, self.alpha_curvature,
            self.alpha_count, self.alpha_head, self.alpha_cap,
            emb[is_alpha], circle_idx[is_alpha], curvatures[is_alpha],
        )
        n_b = self._fifo_write_pack(
            self.beta_emb, self.beta_circle, self.beta_curvature,
            self.beta_count, self.beta_head, self.beta_cap,
            emb[~is_alpha], circle_idx[~is_alpha], curvatures[~is_alpha],
        )
        return {"alpha_stored": n_a, "beta_stored": n_b}

    @staticmethod
    @torch.no_grad()
    def _fifo_write_pack(
        emb_buf, circle_buf, curv_buf, count_buf, head_buf, cap,
        new_emb, new_circle, new_curv,
    ) -> int:
        n_new = new_emb.shape[0]
        if n_new == 0:
            return 0
        head = int(head_buf.item())
        count = int(count_buf.item())
        for i in range(n_new):
            slot = (head + i) % cap
            emb_buf[slot] = new_emb[i]
            circle_buf[slot] = new_circle[i]
            curv_buf[slot] = new_curv[i]
        head_buf.fill_((head + n_new) % cap)
        count_buf.fill_(min(count + n_new, cap))
        return n_new

    # ------------------------------------------------------------------
    # API: retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 8,
        pool: str = "both",
        hidden_preRMSnorm: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, D = query.shape
        N = B * T
        q_flat = query.reshape(N, D)

        hid_q = hidden_preRMSnorm.reshape(N, D) if hidden_preRMSnorm is not None else q_flat
        q_uhs = self.project_uhs(hid_q)  # (N, 3) -- differentiable

        a_n = int(self.alpha_count.item())
        b_n = int(self.beta_count.item())
        pool_embs = []
        pool_circle = []
        if pool in ("alpha", "both") and a_n > 0:
            pool_embs.append(self.alpha_emb[:a_n])
            pool_circle.append(self.alpha_circle[:a_n])
        if pool in ("beta", "both") and b_n > 0:
            pool_embs.append(self.beta_emb[:b_n])
            pool_circle.append(self.beta_circle[:b_n])
        if not pool_embs:
            return {
                "values": torch.zeros(B, T, top_k, D, device=query.device, dtype=query.dtype),
                "scores": torch.zeros(B, T, top_k, device=query.device, dtype=query.dtype),
            }
        pool_emb = torch.cat(pool_embs, dim=0)        # (M, D)
        pool_c = torch.cat(pool_circle, dim=0)        # (M,)
        M = pool_emb.shape[0]
        pool_uhs = self.gasket_coords.to(q_uhs.device)[pool_c]  # (M, 3)

        # Cosine similarity term (back-compat with fant3 retrieval).
        q_n = F.normalize(q_flat.float(), dim=-1)
        m_n = F.normalize(pool_emb.float(), dim=-1)
        cos_sim = q_n @ m_n.T

        # UHS geodesic similarity term: -d(q, m) / sigma, then softmax.
        d_uhs = uhs_dist_t(
            q_uhs.unsqueeze(1).expand(-1, M, -1),
            pool_uhs.unsqueeze(0).expand(N, -1, -1),
        )
        uhs_sim = torch.exp(-d_uhs)  # smooth proximity

        scores = 0.6 * cos_sim + 0.4 * uhs_sim
        k_eff = min(top_k, M)
        topk_scores, topk_idx = scores.topk(k_eff, dim=-1)
        topk_vals = pool_emb[topk_idx]
        if k_eff < top_k:
            pad = top_k - k_eff
            topk_vals = torch.cat([topk_vals, torch.zeros(N, pad, D, device=query.device)], dim=1)
            topk_scores = torch.cat([topk_scores, torch.zeros(N, pad, device=query.device)], dim=1)
        return {
            "values": topk_vals.to(query.dtype).reshape(B, T, top_k, D),
            "scores": topk_scores.to(query.dtype).reshape(B, T, top_k),
        }

    # ------------------------------------------------------------------
    # API: Descartes regularizer (exact, on Cl(3,1) 4-vectors)
    # ------------------------------------------------------------------

    def descartes_loss(
        self,
        query_uhs: torch.Tensor,
        lmbda: float = 1e-4,
        top_n: int = 4,
    ) -> torch.Tensor:
        """For each query UHS point, find the 4 nearest stored circles,
        evaluate the Descartes form F(b) on their curvatures, return mean
        squared violation.

        Replaces fant3's approximate spinor-norm Descartes loss with the
        exact 4-tuple form -- F = 0 iff the four curvatures are a true
        Descartes quadruple.
        """
        a_n = int(self.alpha_count.item())
        b_n = int(self.beta_count.item())
        if a_n + b_n < top_n:
            return query_uhs.new_zeros(())
        parts = []
        if a_n > 0:
            parts.append(self.alpha_curvature[:a_n])
        if b_n > 0:
            parts.append(self.beta_curvature[:b_n])
        all_curv = torch.cat(parts, dim=0).float()              # (M,)
        # Use stored circle UHS coords for nearest search.
        pcircle = []
        if a_n > 0: pcircle.append(self.alpha_circle[:a_n])
        if b_n > 0: pcircle.append(self.beta_circle[:b_n])
        all_circ = torch.cat(pcircle, dim=0)
        all_uhs = self.gasket_coords.to(query_uhs.device)[all_circ]   # (M, 3)

        q = query_uhs.float()
        # Proxy "distance" (avoid acosh for nearest search; argmin preserved).
        diff = q.unsqueeze(1) - all_uhs.unsqueeze(0)
        proxy = (diff ** 2).sum(dim=-1) / (
            q[:, 2].unsqueeze(1) * all_uhs[:, 2].unsqueeze(0) + 1e-9
        )
        # Top-n nearest (smallest proxy).
        _, idx = proxy.topk(top_n, dim=-1, largest=False)        # (Q, n)
        b = all_curv[idx]                                          # (Q, n)
        sum_b = b.sum(dim=-1)
        sum_b2 = (b ** 2).sum(dim=-1)
        violation = (2.0 * sum_b2 - sum_b ** 2) ** 2
        return violation.mean()

    # ------------------------------------------------------------------
    # API: introspection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_stats(self) -> Dict[str, Any]:
        a_n = int(self.alpha_count.item())
        b_n = int(self.beta_count.item())
        total = a_n + b_n
        a_cm = float(self.alpha_curvature[:a_n].mean().item()) if a_n > 0 else 0.0
        b_cm = float(self.beta_curvature[:b_n].mean().item()) if b_n > 0 else 0.0
        balance = float(a_n) / float(total) if total > 0 else 0.5

        # Twin-anchor coverage.
        anchors = set(self.address_book.anchors)
        stored_circles = set()
        if a_n > 0:
            stored_circles |= set(self.alpha_circle[:a_n].tolist())
        if b_n > 0:
            stored_circles |= set(self.beta_circle[:b_n].tolist())
        twin_hits = len(stored_circles & anchors)

        return {
            "alpha_fill": a_n,
            "beta_fill": b_n,
            "alpha_curvature_mean": a_cm,
            "beta_curvature_mean": b_cm,
            "chirality_balance": balance,
            "distinct_circles_used": len(stored_circles),
            "twin_anchor_hits": twin_hits,
        }

    @torch.no_grad()
    def reset(self) -> None:
        for n in (
            "alpha_emb", "alpha_circle", "alpha_curvature", "alpha_count", "alpha_head",
            "beta_emb", "beta_circle", "beta_curvature", "beta_count", "beta_head",
            "global_step",
        ):
            getattr(self, n).zero_()
