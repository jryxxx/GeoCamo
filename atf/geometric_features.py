import torch
import torch.nn.functional as F
from atf.utils import normalize_points_with_margin, compute_curvatures


def extract_geometric_features(
    verts,
    faces,
    normals,
    sampled_points,
    sampled_normals,
    bbox_min=None,
    bbox_max=None,
):
    M = sampled_points.shape[0]
    device = verts.device
    dtype = verts.dtype

    if bbox_min is None or bbox_max is None:
        bbox_min = verts.min(dim=0).values
        bbox_max = verts.max(dim=0).values

    coords_norm = normalize_points_with_margin(
        sampled_points, bbox_min, bbox_max, margin=0.2
    )

    n = F.normalize(sampled_normals, dim=-1)

    mean_curv_all, gauss_curv_all = compute_curvatures(verts, faces)
    dists = torch.cdist(sampled_points, verts)
    nearest_idx = dists.argmin(dim=-1)
    mean_curv = torch.tanh(mean_curv_all[nearest_idx] * 0.01)
    gauss_curv = torch.tanh(gauss_curv_all[nearest_idx] * 0.001)

    ground_y = bbox_min[1]
    bbox_height = (bbox_max[1] - bbox_min[1]).clamp_min(1e-8)
    height = (sampled_points[:, 1] - ground_y) / bbox_height

    centroid = (bbox_min + bbox_max) / 2
    bbox_diag = (bbox_max - bbox_min).norm().clamp_min(1e-8)
    dist_to_centroid = torch.norm(sampled_points - centroid, dim=-1) / bbox_diag

    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    up_dot = (n * up).sum(dim=-1)

    features = torch.stack(
        [
            coords_norm[:, 0],
            coords_norm[:, 1],
            coords_norm[:, 2],
            n[:, 0],
            n[:, 1],
            n[:, 2],
            mean_curv,
            gauss_curv,
            height,
            dist_to_centroid,
            up_dot,
        ],
        dim=-1,
    )

    return features
