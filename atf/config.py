from dataclasses import dataclass
from typing import Tuple


@dataclass
class ATFConfig:
    # =========================================================================
    # Surface sampling
    # =========================================================================
    num_surface_points: int = 4096

    # =========================================================================
    # Hash grid encoder
    # =========================================================================
    # geometry branch: resolution levels (16→512)
    hash_geo_levels: int = 12
    hash_geo_input_dim: int = 3       # geometry branch: input dimension
    # position branch: resolution levels (16→256)
    hash_pos_levels: int = 8
    hash_pos_feat_dim: int = 2        # position branch: features per level
    use_learned_geo_key: bool = True  # learn geo→hash projection via GeoFeatureRouter

    # =========================================================================
    # MLP decoder
    # =========================================================================
    shared_hidden: int = 256
    branch_hidden: int = 128
    # =========================================================================
    # Optimizer
    # =========================================================================
    lr_encoder: float = 1e-3          # encoder (DualHashEncoder)
    lr_decoder: float = 5e-4          # decoder (DualBranchDecoder)
    total_iters: int = 5000

    # =========================================================================
    # Rendering & geometry
    # =========================================================================
    render_size: int = 640            # YOLOv3 input size
    bbox_margin: float = 0.2          # margin for bounding-box normalization
    block_resolution: int = 144        # voxel grid resolution for digital camouflage
    category_temp: float = 0.7        # softmax temperature for palette categories
    subcolor_temp: float = 0.45       # softmax temperature for category subcolors
    local_color_dist_resolution: int = 12  # coarse 3D regions for local color balance

    # =========================================================================
    # Geometric augmentation
    # =========================================================================
    geo_aug_scale: Tuple[float, float] = (0.85, 1.15)
    geo_aug_prob: float = 0.5
