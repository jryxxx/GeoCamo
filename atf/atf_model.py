import torch
import torch.nn as nn

from atf.hash_encoder import DualHashEncoder
from atf.mlp_decoder import DualBranchDecoder


class ATFModel(nn.Module):
    """ATF with geometry-space hash grid encoder.

    Encoder: DualHashEncoder
      - Geometry branch: hash(height, up_dot, curvature) → semantic spatial features
      - Position branch: hash(x, y, z) → spatial reference features
    Decoder: DualBranchDecoder
      - Structure branch: α(p) ∈ [0,1] — attack intensity
      - Color branch: category and subcolor logits over woodland palette
    Output: (alpha, category_logits, subcolor_logits) per surface point
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = DualHashEncoder(
            geo_levels=config.hash_geo_levels,
            pos_levels=config.hash_pos_levels,
            geo_input_dim=config.hash_geo_input_dim,
            pos_feat_dim=config.hash_pos_feat_dim,
            use_learned_geo_key=getattr(config, 'use_learned_geo_key', False),
        )
        decoder_input_dim = self.encoder.total_out_dim + 11  # hash features + raw geo
        self.decoder = DualBranchDecoder(
            input_dim=decoder_input_dim,
            shared_hidden=config.shared_hidden,
            branch_hidden=config.branch_hidden,
        )

    def forward(self, points_01, geo_features):
        hash_feat = self.encoder(points_01, geo_features)
        feat = torch.cat([hash_feat, geo_features], dim=-1)
        return self.decoder(feat)

    def reset_planes(self):
        """No-op for backward compatibility. Hash encoder has no persistent state."""
        pass
