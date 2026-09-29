import torch
import torch.nn.functional as F


WOODLAND_CATEGORIES = [
    "green",
    "earth",
    "shadow",
]

WOODLAND_TARGET_DIST = [0.4, 0.3, 0.3]

WOODLAND_NUM_SUBCOLORS = 48

WOODLAND_PALETTE = [
    [
        [10, 18, 14],
        [14, 24, 18],
        [18, 30, 22],
        [22, 36, 26],
        [28, 44, 30],
        [34, 52, 34],
        [40, 60, 38],
        [48, 68, 42],
        [56, 78, 46],
        [64, 88, 52],
        [74, 98, 58],
        [84, 108, 64],
        [20, 34, 18],
        [28, 46, 22],
        [36, 58, 26],
        [46, 70, 30],
        [58, 82, 34],
        [70, 94, 38],
        [82, 106, 44],
        [94, 118, 52],
        [106, 128, 62],
        [118, 138, 72],
        [130, 148, 84],
        [142, 158, 96],
        [18, 28, 24],
        [26, 40, 32],
        [34, 52, 40],
        [44, 64, 48],
        [54, 76, 56],
        [66, 88, 64],
        [78, 100, 74],
        [92, 112, 84],
        [106, 124, 96],
        [120, 136, 108],
        [134, 148, 120],
        [148, 160, 134],
        [32, 44, 24],
        [42, 56, 28],
        [54, 68, 32],
        [66, 80, 38],
        [78, 92, 44],
        [90, 104, 50],
        [102, 114, 58],
        [114, 124, 68],
        [126, 134, 78],
        [138, 144, 90],
        [150, 154, 104],
        [162, 166, 118],
    ],
    [
        [30, 20, 14],
        [42, 28, 18],
        [54, 36, 22],
        [66, 44, 26],
        [78, 52, 30],
        [90, 60, 36],
        [102, 70, 42],
        [114, 82, 50],
        [126, 94, 60],
        [138, 106, 72],
        [150, 118, 84],
        [162, 132, 98],
        [48, 36, 24],
        [60, 46, 30],
        [72, 56, 36],
        [84, 66, 42],
        [96, 78, 50],
        [108, 90, 58],
        [120, 102, 68],
        [132, 114, 80],
        [144, 126, 92],
        [156, 138, 104],
        [168, 150, 118],
        [180, 162, 132],
        [58, 48, 34],
        [72, 60, 42],
        [86, 72, 50],
        [100, 84, 58],
        [114, 96, 66],
        [128, 108, 76],
        [142, 122, 88],
        [156, 136, 100],
        [170, 150, 112],
        [184, 164, 126],
        [198, 178, 140],
        [212, 192, 156],
        [68, 42, 22],
        [84, 52, 26],
        [100, 64, 32],
        [116, 78, 40],
        [132, 92, 50],
        [148, 108, 62],
        [164, 124, 76],
        [180, 140, 90],
        [196, 156, 106],
        [210, 174, 124],
        [224, 192, 144],
        [238, 210, 166],
    ],
    [
        [8, 10, 10],
        [14, 16, 15],
        [20, 22, 20],
        [28, 30, 27],
        [36, 38, 34],
        [44, 46, 40],
        [52, 54, 48],
        [60, 62, 56],
        [70, 72, 64],
        [82, 84, 76],
        [94, 96, 88],
        [108, 110, 100],
        [12, 18, 20],
        [18, 26, 30],
        [24, 34, 40],
        [32, 44, 52],
        [40, 54, 64],
        [50, 66, 76],
        [62, 78, 88],
        [74, 90, 100],
        [88, 104, 114],
        [102, 118, 128],
        [118, 134, 142],
        [134, 150, 156],
        [14, 20, 16],
        [20, 28, 22],
        [28, 38, 30],
        [36, 48, 38],
        [46, 60, 48],
        [58, 72, 58],
        [70, 84, 70],
        [84, 98, 82],
        [98, 112, 96],
        [114, 128, 110],
        [130, 144, 126],
        [148, 160, 142],
        [28, 24, 20],
        [38, 34, 28],
        [50, 44, 36],
        [62, 56, 46],
        [76, 68, 56],
        [90, 82, 68],
        [104, 96, 82],
        [118, 110, 96],
        [134, 126, 112],
        [150, 142, 128],
        [166, 158, 144],
        [184, 176, 162],
    ],
]


def spatial_hash_3d(ijk):
    return (ijk[:, 0].long() * 73856093) ^ \
           (ijk[:, 1].long() * 19349663) ^ \
           (ijk[:, 2].long() * 83492791)


def get_camo_palette(device):
    return torch.tensor(WOODLAND_PALETTE, dtype=torch.float32, device=device) / 255.0


def get_target_category_dist(device):
    return torch.tensor(WOODLAND_TARGET_DIST, dtype=torch.float32, device=device)


def make_camo_colors_from_grid(uniq_grid, palette, jitter=0.025):
    """Return one deterministic woodland subcolor per finest voxel block."""
    if uniq_grid.shape[0] == 0:
        return palette.reshape(-1, 3)[:0]

    flat_palette = palette.reshape(-1, 3)
    h = spatial_hash_3d(uniq_grid.long())
    idx = (h % flat_palette.shape[0]).long()
    colors = flat_palette[idx]
    if jitter > 0:
        local = (((h % 997).float() / 996.0) - 0.5).unsqueeze(-1)
        colors = colors + local * jitter
    return colors.clamp(0.0, 1.0)


def select_palette_colors(category_logits, subcolor_logits, palette,
                          category_temp=1.0, subcolor_temp=1.0):
    """Softmax-select colors from category/subcolor woodland palette."""
    category_prob = F.softmax(category_logits / category_temp, dim=-1)
    subcolor_prob = F.softmax(subcolor_logits / subcolor_temp, dim=-1)
    weights = category_prob.unsqueeze(-1) * subcolor_prob
    colors = (weights.unsqueeze(-1) * palette.unsqueeze(0)).sum(dim=(1, 2))
    return colors, category_prob, subcolor_prob


def blend_palette_camo(base_colors, alphas, learned_colors):
    return ((1.0 - alphas) * base_colors + alphas * learned_colors).clamp(0.0, 1.0)


def compose_palette_camo(alphas, category_logits, subcolor_logits, uniq_grid,
                         voxel_inv, palette, category_temp=1.0,
                         subcolor_temp=1.0):
    """Shared train/export palette path for voxelized digital camouflage."""
    learned_vox, category_prob, subcolor_prob = select_palette_colors(
        category_logits, subcolor_logits, palette, category_temp, subcolor_temp)
    base_vox = make_camo_colors_from_grid(uniq_grid, palette, jitter=0.0)
    colors = blend_palette_camo(
        base_vox[voxel_inv], alphas[voxel_inv], learned_vox[voxel_inv])
    return colors, category_prob, subcolor_prob


def local_category_distribution_loss(category_logits, target_dist, uniq_grid,
                                     block_resolution, local_resolution=6,
                                     alphas=None, temperature=1.0,
                                     min_weight=1e-4):
    """Match category distribution inside coarse spatial regions.

    Global category matching can still allow one large side panel to become a
    single color category. This groups occupied fine voxels into a coarser 3D
    grid and applies the target distribution per occupied local region.
    """
    if category_logits.shape[0] == 0:
        return category_logits.sum() * 0.0

    local_resolution = max(int(local_resolution), 1)
    block_resolution = max(int(block_resolution), 1)
    category_prob = F.softmax(category_logits / temperature, dim=-1)

    local_grid = torch.div(
        uniq_grid.long() * local_resolution,
        block_resolution,
        rounding_mode="floor",
    ).clamp(0, local_resolution - 1)
    local_id = local_grid[:, 0] * local_resolution * local_resolution \
        + local_grid[:, 1] * local_resolution + local_grid[:, 2]
    n_regions = local_resolution ** 3

    if alphas is None:
        weights = torch.ones(category_logits.shape[0], 1,
                             device=category_logits.device,
                             dtype=category_logits.dtype)
    else:
        weights = alphas.detach().clamp_min(0.0)

    weighted_prob = weights * category_prob
    sums = torch.zeros(n_regions, category_prob.shape[-1],
                       device=category_logits.device,
                       dtype=category_logits.dtype)
    denom = torch.zeros(n_regions, 1, device=category_logits.device,
                        dtype=category_logits.dtype)
    sums.index_add_(0, local_id, weighted_prob)
    denom.index_add_(0, local_id, weights)

    valid = denom.squeeze(-1) > min_weight
    if not torch.any(valid):
        return category_logits.sum() * 0.0

    observed = (sums[valid] / denom[valid].clamp_min(1e-6)).clamp_min(1e-8)
    target = (target_dist / target_dist.sum()).unsqueeze(0).expand_as(observed)
    return F.kl_div(observed.log(), target, reduction="batchmean")
