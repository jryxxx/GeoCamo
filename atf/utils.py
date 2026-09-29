import torch
import torch.nn.functional as F
from pytorch3d.ops import sample_points_from_meshes


def compute_mesh_bbox(verts):
    bbox_min = verts.min(dim=0).values
    bbox_max = verts.max(dim=0).values
    return bbox_min, bbox_max


def normalize_points(points, bbox_min, bbox_max):
    scale = bbox_max - bbox_min
    scale = torch.where(scale < 1e-8, torch.ones_like(scale), scale)
    return (points - bbox_min) / scale


def normalize_points_with_margin(points, bbox_min, bbox_max, margin=0.2):
    size = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2
    expanded_size = size * (1.0 + 2.0 * margin)
    normalized = (points - (center - expanded_size / 2)) / expanded_size
    return normalized.clamp(0.0, 1.0)


def sample_surface_area_weighted(mesh, n_points):
    sampled = sample_points_from_meshes(mesh, n_points, return_normals=True)
    if isinstance(sampled, tuple):
        points, normals = sampled
    else:
        points = sampled
        normals = None
    return points.squeeze(0), normals.squeeze(0) if normals is not None else None


def get_vertex_normals(mesh):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    face_normals = torch.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]],
        verts[faces[:, 2]] - verts[faces[:, 0]],
        dim=-1,
    )
    face_normals = F.normalize(face_normals, dim=-1)
    vert_normals = torch.zeros_like(verts)
    vert_normals.index_add_(0, faces[:, 0], face_normals)
    vert_normals.index_add_(0, faces[:, 1], face_normals)
    vert_normals.index_add_(0, faces[:, 2], face_normals)
    return F.normalize(vert_normals, dim=-1)


def stretch_mesh(verts, scale_factor):
    if isinstance(scale_factor, (tuple, list)):
        lo, hi = scale_factor
    else:
        lo, hi = 1.0 / scale_factor, scale_factor
    scales = lo + (hi - lo) * torch.rand(3, device=verts.device)
    center = (verts.min(dim=0).values + verts.max(dim=0).values) / 2
    return (verts - center) * scales + center


def compute_face_centroids(verts, faces):
    """Compute face centroids and face normals.

    Args:
        verts: (N, 3)
        faces: (F, 3) face indices

    Returns:
        centroids: (F, 3) per-face centroid positions
        face_normals: (F, 3) per-face unit normals
    """
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    face_normals = F.normalize(torch.cross(v1 - v0, v2 - v0), dim=-1)
    return centroids, face_normals


def compute_curvatures(verts, faces):
    from pytorch3d.ops import cot_laplacian

    L, inv_areas = cot_laplacian(verts, faces)
    inv_a = inv_areas.squeeze(-1)  # (N,)

    Lv = L.mm(verts)
    mean_curv_vec = 0.5 * inv_a.unsqueeze(1) * Lv
    mean_curv = torch.norm(mean_curv_vec, dim=-1)

    N = verts.shape[0]
    gauss_curv = torch.zeros(N, device=verts.device, dtype=verts.dtype)

    for i in range(3):
        i0, i1, i2 = i, (i + 1) % 3, (i + 2) % 3
        v0 = verts[faces[:, i0]]
        v1 = verts[faces[:, i1]]
        v2 = verts[faces[:, i2]]
        e1 = F.normalize(v1 - v0, dim=-1)
        e2 = F.normalize(v2 - v0, dim=-1)
        cos_angle = (e1 * e2).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cos_angle)
        gauss_curv.index_add_(0, faces[:, i0], angle)

    gauss_curv = (2.0 * torch.pi - gauss_curv) * inv_a
    return mean_curv, gauss_curv


# ==============================================================================
# Voxel quantization for block-wise ATF query
# ==============================================================================

def voxelize_surface_points(points, bbox_min, bbox_max, resolution):
    """Quantize 3D surface points into voxel grid, returning representatives.

    All points within the same voxel cell share one ATF query, producing
    digital-camouflage-style block patterns. The voxel grid is defined
    uniformly within the vehicle's bounding box.

    Args:
        points:    (M, 3) surface points in world space
        bbox_min:  (3,) bounding box minimum
        bbox_max:  (3,) bounding box maximum
        resolution: int, cells per longest axis

    Returns:
        first_idx:  (U,) indices of representative points (one per occupied voxel)
        inverse:    (M,) mapping from original point index to voxel index
        n_voxels:   int, number of unique occupied voxels
    """
    device = points.device
    M = points.shape[0]

    size = (bbox_max - bbox_min).clamp_min(1e-8)
    voxel_size = size.max() / resolution
    axis_resolution = torch.ceil(size / voxel_size).long().clamp_min(1)
    grid = torch.floor((points - bbox_min) / voxel_size).long()
    grid = torch.maximum(grid, torch.zeros_like(grid))
    grid = torch.minimum(grid, axis_resolution.unsqueeze(0) - 1)

    flat = grid[:, 0] * axis_resolution[1] * axis_resolution[2] \
        + grid[:, 1] * axis_resolution[2] \
        + grid[:, 2]  # (M,) 1D voxel index

    # Sort → diff boundaries → first point per voxel
    sorted_flat, sort_idx = flat.sort()
    changes = torch.cat([
        torch.tensor([True], device=device),
        sorted_flat[1:] != sorted_flat[:-1]
    ])
    first_idx = sort_idx[changes]  # (U,)

    # Inverse mapping: original point → voxel index
    inverse = torch.zeros(M, dtype=torch.long, device=device)
    inverse[sort_idx] = changes.long().cumsum(0) - 1  # (U,) → (M,)

    # Unique voxel grid indices (for deterministic palette assignment)
    uniq_grid = grid[first_idx]  # (U, 3)

    return first_idx, inverse, first_idx.shape[0], uniq_grid


# ==============================================================================
# Natural camouflage palette
# ==============================================================================

def get_natural_palette(device):
    """Base palette — natural camo colors (olive, brown, gray, slate)."""
    palette = torch.tensor(
        [
            [44, 72, 56],    # dark olive
            [62, 92, 68],    # olive green
            [86, 112, 78],   # light olive
            [98, 82, 60],    # warm brown
            [118, 106, 84],  # light brown
            [136, 120, 94],  # sand
            [72, 78, 84],    # dark gray
            [96, 98, 96],    # mid gray
            [122, 124, 116], # light gray
            [52, 64, 72],    # slate
        ],
        dtype=torch.float32,
        device=device,
    )
    return palette / 255.0


def get_dense_palette(base_palette, max_colors=48):
    """Expand base palette with brightness variants and blends."""
    variants = [base_palette]
    # Brightness variants
    for scale in (0.82, 0.94, 1.06, 1.18):
        variants.append((base_palette * scale).clamp(0.0, 1.0))
    # Color blends (cyclic shift + alpha blend)
    rolled = torch.roll(base_palette, shifts=-1, dims=0)
    for alpha in (0.25, 0.5, 0.75):
        variants.append(
            ((1.0 - alpha) * base_palette + alpha * rolled).clamp(0.0, 1.0))

    dense = torch.cat(variants, dim=0)
    dense_u8 = (dense * 255.0).round().to(torch.int64)
    dense = torch.unique(dense_u8, dim=0).float() / 255.0
    if dense.shape[0] > max_colors:
        dense = dense[:max_colors]
    return dense.to(base_palette.device)


def spatial_hash_3d(ijk):
    """Hash 3D integer grid coordinates for deterministic palette assignment.

    Same grid cell → same hash → same palette color, ensuring consistency
    across training steps, augmentations, and migration.
    """
    return (ijk[:, 0].long() * 73856093) ^ \
           (ijk[:, 1].long() * 19349663) ^ \
           (ijk[:, 2].long() * 83492791)


def sample_palette_colors(uniq_grid, palette, camo_green):
    """Assign palette colors to voxels based on spatial hash.

    Args:
        uniq_grid: (U, 3) long tensor, voxel grid indices [0, resolution)
        palette: (P, 3) dense color palette
        camo_green: (3,) fallback color

    Returns:
        voxel_base: (U, 3) palette-assigned colors
    """
    if uniq_grid.shape[0] == 0:
        return palette[:0]
    hashes = spatial_hash_3d(uniq_grid)
    idx = (hashes % palette.shape[0]).long()
    return palette[idx]
