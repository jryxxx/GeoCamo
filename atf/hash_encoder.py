"""Multi-resolution hash grid encoder operating in geometric feature space.

The key difference from standard hash grids (e.g. Instant-NGP, TT3D):
- Standard: hash(x, y, z) → features (encodes spatial position)
- Ours:   hash(height, up_dot, curvature) → features (encodes geometric semantics)

This allows cross-vehicle generalization: "upward-facing, low-curvature surface
at ~1m height" maps to the same hash buckets regardless of vehicle model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fast_hash_3d(ijk, hash_size):
    """Spatial hash for 3D integer coordinates. Same principle as Instant-NGP."""
    i, j, k = ijk[..., 0].long(), ijk[..., 1].long(), ijk[..., 2].long()
    h = (i * 19349663) ^ (j * 96925573) ^ (k * 76923077)
    return h % hash_size


class MultiResHashGrid(nn.Module):
    """Multi-resolution hash grid encoder in geometric feature subspace.

    Input: (M, 3) geometric features mapped to [0, 1]^3
    Output: (M, n_levels * n_features)

    Uses trilinear interpolation within each level's hash grid.
    """

    def __init__(self, input_dim=3, n_levels=12, n_features=2,
                 base_res=16, finest_res=512, hash_size=2**19):
        super().__init__()
        self.input_dim = input_dim
        self.n_levels = n_levels
        self.n_features = n_features
        self.base_res = base_res
        self.hash_size = hash_size

        self.per_level_scale = np.exp2(
            np.log2(finest_res / base_res) / max(n_levels - 1, 1)
        )

        self.tables = nn.ParameterList()
        for i in range(n_levels):
            res = int(base_res * self.per_level_scale ** i)
            # Number of hash entries for this level
            n_entries = min(hash_size, max(res, 1) ** input_dim)
            n_entries = (n_entries // 8) * 8  # Align to 8 for CUDA efficiency
            self.tables.append(nn.Parameter(
                torch.empty(n_entries, n_features)
            ))

        self.output_dim = n_levels * n_features
        self._register_load_state_dict_pre_hook(self._reshape_on_load)
        self.reset_parameters()

    def _reshape_on_load(self, state_dict, prefix, *args):
        """Reshape hash tables if checkpoint shape differs (e.g. config change)."""
        for i in range(self.n_levels):
            key = f"{prefix}tables.{i}"
            if key in state_dict:
                if state_dict[key].shape != self.tables[i].shape:
                    new_table = torch.empty_like(self.tables[i].data)
                    nn.init.uniform_(new_table, -1e-4, 1e-4)
                    # Copy overlapping region
                    min0 = min(state_dict[key].shape[0], new_table.shape[0])
                    min1 = min(state_dict[key].shape[1], new_table.shape[1])
                    new_table[:min0, :min1] = state_dict[key][:min0, :min1]
                    state_dict[key] = new_table

    def reset_parameters(self):
        for t in self.tables:
            nn.init.uniform_(t, -1e-4, 1e-4)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (M, input_dim) query points in [0, 1]^input_dim

        Returns:
            features: (M, n_levels * n_features)
        """
        M = x.shape[0]
        device = x.device
        dtype = x.dtype
        all_features = []

        for lvl, table in enumerate(self.tables):
            res = int(self.base_res * self.per_level_scale ** lvl)
            if res < 1:
                res = 1

            scaled = x * (res - 1 + 1e-6)  # (M, D)
            floor_coord = torch.floor(scaled).long().clamp(0, res - 1)
            ceil_coord = torch.clamp(floor_coord + 1, 0, res - 1)
            frac = (scaled - floor_coord.float()).to(dtype)

            level_feat = torch.zeros(M, self.n_features, device=device, dtype=dtype)

            # Trilinear interpolation: iterate 2^input_dim corners
            for corner_idx in range(2 ** self.input_dim):
                corner_mask = [(corner_idx >> d) & 1 for d in range(self.input_dim)]
                corner_coord = torch.stack([
                    floor_coord[:, d] if m == 0 else ceil_coord[:, d]
                    for d, m in enumerate(corner_mask)
                ], dim=-1)  # (M, D)

                weight = torch.ones(M, 1, device=device, dtype=dtype)
                for d, m in enumerate(corner_mask):
                    w_d = (1.0 - frac[:, d]) if m == 0 else frac[:, d]
                    weight = weight * w_d.unsqueeze(-1)

                h = fast_hash_3d(corner_coord, table.shape[0])
                level_feat = level_feat + table[h] * weight

            all_features.append(level_feat)

        return torch.cat(all_features, dim=-1)


class DualHashEncoder(nn.Module):
    """Dual-branch hash encoder: geometry branch + position branch.

    Geometry branch maps geometric semantics to hash features:
      - Hand-picked (default): hash(height, up_dot, curvature)
      - Learned (--use-learned-geo-key): GeoFeatureRouter + hash

    Position branch: hash(x, y, z) → spatial features.

    Cross-vehicle generalization comes from the geometry branch: same
    geometric structure maps to same hash buckets regardless of vehicle.
    """

    def __init__(self, geo_levels=12, pos_levels=8, geo_input_dim=3, pos_feat_dim=2,
                 geo_base=16, geo_finest=512, pos_base=16, pos_finest=256,
                 hash_size=2**19, use_learned_geo_key=False):
        super().__init__()

        self.use_learned_geo_key = use_learned_geo_key

        if use_learned_geo_key:
            from atf.geo_router import GeoFeatureRouter
            self.geo_router = GeoFeatureRouter(K=geo_input_dim)

        self.geo_encoder = MultiResHashGrid(
            input_dim=geo_input_dim, n_levels=geo_levels, n_features=2,
            base_res=geo_base, finest_res=geo_finest, hash_size=hash_size
        )

        self.pos_encoder = MultiResHashGrid(
            input_dim=3, n_levels=pos_levels, n_features=pos_feat_dim,
            base_res=pos_base, finest_res=pos_finest, hash_size=hash_size
        )

        self.geo_out_dim = self.geo_encoder.output_dim
        self.pos_out_dim = self.pos_encoder.output_dim
        self.total_out_dim = self.geo_out_dim + self.pos_out_dim

    def forward(self, points_01, geo_features):
        """Forward pass.

        Args:
            points_01: (M, 3) normalized 3D coordinates in [0,1]^3
            geo_features: (M, 11) raw geometric features

        Returns:
            encoded: (M, total_out_dim) concatenated hash features
        """
        # Geometry branch: project geometric features to hash key
        if self.use_learned_geo_key:
            geo_key = self.geo_router(geo_features)  # (M, K) in [0,1]
        else:
            geo_key = torch.stack([
                geo_features[:, 8],   # height
                geo_features[:, 10],  # up direction dot product
                geo_features[:, 6],   # mean curvature (tanh-scaled, in [-1,1])
            ], dim=-1)
            geo_key[:, 2] = (geo_key[:, 2] + 1.0) / 2.0
            geo_key = geo_key.clamp(0.0, 1.0)

        geo_hash = self.geo_encoder(geo_key)

        # Position branch: spatial coordinates
        pos_hash = self.pos_encoder(points_01.clamp(0.0, 1.0))

        return torch.cat([geo_hash, pos_hash], dim=-1)

    def reset_planes(self):
        """No-op for backward compatibility. Hash encoder has no persistent planes."""
        pass
