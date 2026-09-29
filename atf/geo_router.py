"""Learned geometric feature projection for hash grid encoding.

Instead of hand-picking 3 of 11 geometric features as the hash key
(height, up_dot, curvature), this module learns to project and combine
features from three semantic groups with explicit pairwise interactions.

Semantic decomposition:
  Surface group: normal(nx,ny,nz), mean_curv, gauss_curv, up_dot — "shape"
  Spatial group: height, dist_to_centroid — "position in vehicle body"
  Global group:  coords_norm(x,y,z) — "context in bounding box"

Pairwise Hadamard products between group features capture composite
geometric patterns like "vertical surface at medium height = door panel".
This is the simplest form of bilinear interaction — zero extra parameters
and each product dimension has a direct physical interpretation.
"""

import torch
import torch.nn as nn


class GeoFeatureRouter(nn.Module):
    """Learned geometric feature projector with pairwise group interactions.

    Args:
        K: output hash key dimension (default 3, matching current config)
        d: unified feature dimension per branch after projection

    Input:  (M, 11) raw geometric features
    Output: (M, K) hash key in [0, 1]^K
    """

    def __init__(self, K=3, d=16):
        super().__init__()

        # Surface branch: "what shape is this?"
        # Input: normal(3) + mean_curv(1) + gauss_curv(1) + up_dot(1) = 6
        self.enc_surface = nn.Sequential(
            nn.Linear(6, 32), nn.LayerNorm(32), nn.ReLU(inplace=True),
            nn.Linear(32, d), nn.LayerNorm(d), nn.ReLU(inplace=True),
        )

        # Spatial branch: "where is this on the vehicle?"
        # Input: height(1) + dist_to_centroid(1) = 2
        self.enc_spatial = nn.Sequential(
            nn.Linear(2, 16), nn.LayerNorm(16), nn.ReLU(inplace=True),
            nn.Linear(16, d), nn.LayerNorm(d), nn.ReLU(inplace=True),
        )

        # Global branch: "where in the bounding box?"
        # Input: coords_norm(3) = 3
        self.enc_global = nn.Sequential(
            nn.Linear(3, 16), nn.LayerNorm(16), nn.ReLU(inplace=True),
            nn.Linear(16, d), nn.LayerNorm(d), nn.ReLU(inplace=True),
        )

        # Fusion: 3 base + 3 pairwise interactions = 6*d dimensions
        self.fusion = nn.Sequential(
            nn.Linear(d * 6, 32), nn.ReLU(inplace=True),
            nn.Linear(32, K), nn.Sigmoid(),
        )

    def forward(self, geo_features):
        """Forward pass.

        Args:
            geo_features: (M, 11) tensor with columns:
                [0:3]   coords_norm    (x, y, z)
                [3:6]   normal         (nx, ny, nz)
                [6]     mean_curvature (tanh-scaled, [-1,1])
                [7]     gauss_curvature (tanh-scaled, [-1,1])
                [8]     height         (normalized, [0,1])
                [9]     dist_to_centroid (normalized, [0,1])
                [10]    up_dot         (n·up, [-1,1])

        Returns:
            geo_key: (M, K) hash key in [0, 1]^K
        """
        # ── Semantic group encoding ──

        surf_feat = torch.cat([
            geo_features[:, 3:6],    # normal (3)
            geo_features[:, 6:7],    # mean curvature (1)
            geo_features[:, 7:8],    # gauss curvature (1)
            geo_features[:, 10:11],  # up dot product (1)
        ], dim=-1)  # (M, 6)

        spat_feat = torch.cat([
            geo_features[:, 8:9],    # height (1)
            geo_features[:, 9:10],   # distance to centroid (1)
        ], dim=-1)  # (M, 2)

        glob_feat = geo_features[:, 0:3]  # (M, 3)

        surf = self.enc_surface(surf_feat)  # (M, d)
        spat = self.enc_spatial(spat_feat)  # (M, d)
        glob = self.enc_global(glob_feat)   # (M, d)

        # ── Pairwise Hadamard interactions ──
        # Each product captures composite patterns:
        #   surf ⊙ spat → "this shape at this position"
        #   surf ⊙ glob → "this shape in this region"
        #   spat ⊙ glob → "this position in this region"

        surf_spat = surf * spat
        surf_glob = surf * glob
        spat_glob = spat * glob

        # ── Fuse and project ──
        fused = torch.cat(
            [surf, spat, glob, surf_spat, spat_glob, surf_glob], dim=-1
        )  # (M, 6*d)

        geo_key = self.fusion(fused)  # (M, K)
        return geo_key
