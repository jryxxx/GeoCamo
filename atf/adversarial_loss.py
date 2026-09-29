"""Softmax-weighted top-K detector score loss used during field training."""

import torch
import torch.nn.functional as F


class APAdversarialLoss:
    """Multi-scale adversarial loss targeting the score distribution behind AP50.

    softmax(s / τ) · s  over the top-K anchors from all YOLOv3 detection
    scales.  Range [0, 1], no intrinsic floor.

    Parameters
    ----------
    temperature : float
        Softmax temperature.  Lower → sharper focus on the max.
        Default 0.1 for scores ∈ [0, 1].
    topk : int
        Number of highest-scoring anchors to include.  Default 5.
    """

    def __init__(self, temperature: float = 0.1, topk: int = 5):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if topk < 1:
            raise ValueError("topk must be >= 1")
        self.temperature = temperature
        self.topk = topk

    # ------------------------------------------------------------------
    # Public API — compatible with ComputeLoss.__call__(p, targets, target_class)
    # ------------------------------------------------------------------
    def __call__(
        self,
        p: "list[torch.Tensor] | torch.Tensor",
        targets: torch.Tensor,                # unused — kept for API compatibility
        target_class: "int | None" = 2,
    ) -> torch.Tensor:
        """Compute the AP-aware adversarial loss.

        Parameters
        ----------
        p : list of Tensor or Tensor
            YOLOv3 training-mode output: list of ``(B, na, gh, gw, 5+nc)``
            tensors, one per detection scale (typically 3 scales).
        targets : Tensor
            Detection targets.  **Not used by this loss** — present only for
            API compatibility with ``ComputeLoss``.
        target_class : int or None
            Class index to suppress.  ``None`` suppresses the max class per
            anchor (class-agnostic attack).

        Returns
        -------
        Tensor
            Scalar loss.  Higher → detector more confident → attacker wants
            to minimise this.
        """
        device = p[0].device if isinstance(p, (list, tuple)) else p.device

        # ── Normalise to list of scales ──────────────────────────────
        p_list = list(p) if isinstance(p, (list, tuple)) else [p]

        # ── Collect target-class scores from every scale ─────────────
        all_scores: list[torch.Tensor] = []
        for pi in p_list:
            if pi.numel() == 0:
                continue

            obj_conf = pi[..., 4]                           # (B, na, gh, gw)
            cls_conf = pi[..., 5:]                          # (B, na, gh, gw, nc)

            if cls_conf.shape[-1] == 0:
                score = obj_conf
            elif target_class is None:
                score = obj_conf * cls_conf.max(dim=-1).values
            else:
                cls_idx = max(0, min(int(target_class), cls_conf.shape[-1] - 1))
                score = obj_conf * cls_conf[..., cls_idx]

            all_scores.append(score.flatten())

        if not all_scores:
            return torch.tensor(0.0, device=device)

        scores = torch.cat(all_scores)                      # (N_total,)

        # ── Top-K: exclude the long tail of near-zero anchors ────────
        K = min(self.topk, int(scores.numel()))
        top_k, _ = torch.topk(scores, K)

        # ── Softmax-weighted mean ────────────────────────────────────
        #   softmax(s / τ) · s   ∈ [0, max(s)].
        #
        #   τ = 0.1, K = 5:
        #     One anchor at 0.95  → loss ≈ 0.95   (gradient focused)
        #     All 5 equal at 0.5  → loss = 0.5    (gradient uniform)
        #     All at 0            → loss = 0       (attack succeeded)
        #
        #   No log(K)·τ floor — loss cleanly reflects detection strength.
        weights = F.softmax(top_k / self.temperature, dim=0)
        return (weights * top_k).sum()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"APAdversarialLoss(temperature={self.temperature}, topk={self.topk})"
