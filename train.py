

from atf.renderer import NvdiffrastRenderer
from atf.atf_model import ATFModel
from atf.geometric_features import extract_geometric_features
from atf.losses import cross_vehicle_consistency_loss
from atf.utils import (
    compute_mesh_bbox,
    normalize_points_with_margin,
    get_vertex_normals,
    stretch_mesh,
    voxelize_surface_points,
)
from atf.camo import (
    get_camo_palette,
    compose_palette_camo,
    local_category_distribution_loss,
    get_target_category_dist,
    WOODLAND_CATEGORIES,
)
from atf.config import ATFConfig
from atf.adversarial_loss import APAdversarialLoss
from utils.torch_utils import intersect_dicts
from models.yolo import Model
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from tqdm import tqdm
from PIL import Image
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import json
import os
import random
from pathlib import Path
from dataclasses import asdict


# ==============================================================================
# Utility functions
# ==============================================================================

def load_yolov3(weights_path, cfg_path, data_yaml, device):
    with open(data_yaml) as f:
        data_dict = yaml.safe_load(f)
    nc = int(data_dict["nc"])

    hyp_path = Path(__file__).resolve().parent / "data" / "hyp.scratch.yaml"
    with open(hyp_path) as f:
        hyp = yaml.safe_load(f)

    ckpt = torch.load(weights_path, map_location=device)
    model = Model(
        cfg_path or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")
    ).to(device)

    exclude = ["anchor"] if (cfg_path or hyp.get("anchors")) else []
    state_dict = ckpt["model"].float().state_dict()
    state_dict = intersect_dicts(
        state_dict, model.state_dict(), exclude=exclude)
    model.load_state_dict(state_dict, strict=False)

    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    gs = max(int(model.stride.max()), 32)
    hyp["box"] *= 3.0 / model.model[-1].nl
    hyp["cls"] *= nc / 80.0 * 3.0 / model.model[-1].nl
    hyp["obj"] *= (640 / 640) ** 2 * 3.0 / model.model[-1].nl
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.names = data_dict["names"]
    return model


def find_map_kd_texture(obj_path):
    obj_path = Path(obj_path)
    obj_dir = obj_path.parent
    mtl_paths = []
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0] == "mtllib":
                mtl_paths.append(obj_dir / parts[1])

    for mtl_path in mtl_paths:
        if not mtl_path.exists():
            continue
        with open(mtl_path, "r", errors="ignore") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2 and parts[0] == "map_Kd":
                    tex_path = Path(parts[1])
                    if not tex_path.is_absolute():
                        tex_path = mtl_path.parent / tex_path
                    return tex_path.resolve()
    return None


def load_texture_image(texture_path, device):
    if texture_path is None:
        return None
    if not texture_path.exists():
        raise FileNotFoundError(f"Texture not found: {texture_path}")
    img = Image.open(texture_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def sample_texture(texture, uv):
    if texture is None or uv is None:
        return None
    grid = torch.empty(1, uv.shape[0], 1, 2,
                       device=uv.device, dtype=texture.dtype)
    grid[0, :, 0, 0] = uv[:, 0].clamp(0.0, 1.0) * 2.0 - 1.0
    grid[0, :, 0, 1] = 1.0 - uv[:, 1].clamp(0.0, 1.0) * 2.0
    sampled = F.grid_sample(
        texture.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0).squeeze(-1).T
    return sampled.contiguous()


def load_mesh(obj_path, device):
    verts, faces, aux = load_obj(obj_path)
    verts_uv = aux.verts_uvs.to(device) if aux.verts_uvs is not None else None
    faces_uv = faces.textures_idx.to(
        device) if faces.textures_idx is not None else None
    if verts_uv is not None and verts_uv.numel() == 0:
        verts_uv = None
    if faces_uv is not None and faces_uv.numel() == 0:
        faces_uv = None
    texture_path = find_map_kd_texture(obj_path)
    texture = load_texture_image(texture_path, device)
    return verts.to(device), faces.verts_idx.to(device), verts_uv, faces_uv, texture, texture_path


def save_checkpoint(model, optimizer, iteration, config, path, epoch=None, metrics=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if epoch is not None:
        payload["epoch"] = epoch
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, path)


def save_training_params(args, config, vehicles, faces_paths, num_samples,
                         iters_per_epoch, device, out_dir):
    metadata = {
        "cli_args": vars(args),
        "effective_config": asdict(config),
        "derived": {
            "num_samples": num_samples,
            "iters_per_epoch": iters_per_epoch,
            "total_iters": config.total_iters,
            "n_vehicles": len(vehicles),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None,
        },
        "vehicles": [
            {
                "name": v["name"],
                "obj_file": args.obj_file[i],
                "faces_file": faces_paths[i],
                "n_verts": int(v["verts"].shape[0]),
                "n_faces": int(v["n_faces"]),
                "n_optimizable_faces": int(
                    len(v["face_list"]) if v["face_list"] is not None
                    else v["n_faces"]),
            }
            for i, v in enumerate(vehicles)
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(Path(out_dir) / "training_params.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_image(tensor, path):
    arr = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def save_image_with_boxes(tensor, targets, path):
    arr = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    if targets is not None and targets.numel() > 0:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        w, h = img.size
        line_w = max(2, min(w, h) // 200)
        for target in targets.detach().cpu():
            cls_id = int(target[1].item())
            xc, yc, bw, bh = target[2:6].tolist()
            x1 = max(0, min(w - 1, int(round((xc - bw / 2.0) * w))))
            y1 = max(0, min(h - 1, int(round((yc - bh / 2.0) * h))))
            x2 = max(0, min(w - 1, int(round((xc + bw / 2.0) * w))))
            y2 = max(0, min(h - 1, int(round((yc + bh / 2.0) * h))))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_w)
            draw.text((x1, max(0, y1 - 14)), str(cls_id), fill=(255, 0, 0))
    img.save(path)


def save_palette_visualization(palette, target_dist, path):
    palette_u8 = (palette.detach().cpu().clamp(
        0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    target = target_dist.detach().cpu().numpy()
    rows, cols = palette_u8.shape[:2]
    swatch = 56
    label_w = 128
    dist_w = 72
    margin = 16
    gap = 8
    width = margin * 2 + label_w + cols * swatch + dist_w
    height = margin * 2 + rows * swatch + (rows - 1) * gap
    img = Image.new("RGB", (width, height), (245, 245, 245))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    for r, name in enumerate(WOODLAND_CATEGORIES):
        y = margin + r * (swatch + gap)
        draw.text((margin, y + swatch // 2 - 7), name, fill=(20, 20, 20))
        for c in range(cols):
            x = margin + label_w + c * swatch
            color = tuple(int(v) for v in palette_u8[r, c])
            draw.rectangle([x, y, x + swatch - 1, y + swatch - 1], fill=color)
            draw.rectangle([x, y, x + swatch - 1, y +
                           swatch - 1], outline=(32, 32, 32))
        draw.text(
            (margin + label_w + cols * swatch + 12, y + swatch // 2 - 7),
            f"{target[r]:.2f}",
            fill=(20, 20, 20),
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def resolve_dataset_paths(args):
    with open(args.data) as f:
        data_dict = yaml.safe_load(f) or {}

    npz_dir = args.npz_dir or data_dict.get("npz_dir")
    label_dir = args.label_dir or data_dict.get("label_dir")
    mask_dir = args.mask_dir or data_dict.get("mask_dir")
    npz_color_order = args.npz_color_order or data_dict.get(
        "npz_color_order", "bgr")

    if npz_dir is None:
        root = data_dict.get("path")
        train_path = data_dict.get("train")
        if root and train_path:
            npz_dir = str(Path(root) / train_path)

    if npz_dir is None:
        raise ValueError(
            "Dataset npz_dir is required in --npz-dir or data yaml")
    return npz_dir, label_dir, mask_dir, npz_color_order.lower()


def load_dataset(npz_dir, label_dir=None, mask_dir=None):
    npz_paths = sorted(Path(npz_dir).glob("*.npz"))
    items = []
    for npz_path in npz_paths:
        label_path = None
        if label_dir:
            label_path = os.path.join(label_dir, npz_path.stem + ".txt")
        items.append((str(npz_path), label_path))
    if not items:
        raise FileNotFoundError(f"No NPZ files found in {npz_dir}")
    return items, mask_dir


def letterbox_image(img, image_size, pad_value=0, interpolation=cv2.INTER_LINEAR):
    h0, w0 = img.shape[:2]
    scale = image_size / max(h0, w0)
    new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
    if (new_w, new_h) != (w0, h0):
        img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    if img.ndim == 2:
        canvas = np.full((image_size, image_size), pad_value, dtype=img.dtype)
    else:
        canvas = np.full(
            (image_size, image_size, img.shape[2]), pad_value, dtype=img.dtype)
    pad_x = (image_size - new_w) // 2
    pad_y = (image_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img
    return canvas, scale, pad_x, pad_y, w0, h0


def load_npz_data(npz_path, label_path, mask_dir, image_size, device,
                  inferred_class=2, color_order="bgr"):
    data = np.load(npz_path, allow_pickle=True)
    img = data["img"]
    cam_trans = data["cam_trans"]

    img, scale, pad_x, pad_y, w0, h0 = letterbox_image(
        img, image_size, pad_value=114, interpolation=cv2.INTER_LINEAR)
    img = img.transpose((2, 0, 1))
    if color_order == "bgr":
        img = img[::-1]
    img = np.ascontiguousarray(img)
    background = torch.from_numpy(img).float().to(device)
    if background.max() > 1.5:
        background = background / 255.0

    mask = None
    if mask_dir:
        mask_path = os.path.join(mask_dir, os.path.basename(
            npz_path).replace(".npz", ".png"))
        mask = cv2.imread(mask_path)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        mask, _, _, _, _, _ = letterbox_image(
            mask, image_size, pad_value=0, interpolation=cv2.INTER_NEAREST)
        mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
    if mask is not None:
        mask = torch.from_numpy(mask.astype("float32")).to(device)

    targets = None
    if label_path and os.path.exists(label_path):
        labels = np.loadtxt(label_path, ndmin=2)
        if len(labels) > 0:
            nl = len(labels)
            targets = torch.zeros((nl, 6), device=device)
            targets[:, 1:] = torch.from_numpy(labels).float().to(device)
            targets[:, 2] = (targets[:, 2] * w0 * scale + pad_x) / image_size
            targets[:, 3] = (targets[:, 3] * h0 * scale + pad_y) / image_size
            targets[:, 4] = targets[:, 4] * w0 * scale / image_size
            targets[:, 5] = targets[:, 5] * h0 * scale / image_size
            targets[:, 2:6] = targets[:, 2:6].clamp(0.0, 1.0)

    return background, mask, targets, cam_trans


def mask_to_target(mask, device, class_id=2):
    if mask is None:
        return torch.zeros((0, 6), device=device)
    if mask.ndim == 3:
        mask_2d = mask.squeeze(0)
    else:
        mask_2d = mask
    rows, cols = torch.where(mask_2d > 0.5)
    if rows.numel() == 0 or cols.numel() == 0:
        return torch.zeros((0, 6), device=device)

    h, w = mask_2d.shape
    x_min = cols.min().float()
    x_max = cols.max().float() + 1.0
    y_min = rows.min().float()
    y_max = rows.max().float() + 1.0

    target = torch.zeros((1, 6), device=device)
    target[0, 1] = float(class_id)
    target[0, 2] = ((x_min + x_max) * 0.5) / float(w)
    target[0, 3] = ((y_min + y_max) * 0.5) / float(h)
    target[0, 4] = (x_max - x_min) / float(w)
    target[0, 5] = (y_max - y_min) / float(h)
    target[:, 2:6] = target[:, 2:6].clamp(0.0, 1.0)
    return target


def remap_obj_faces_to_triangulated(obj_path, face_list, n_tri_faces):
    """Map OBJ face-line ids to PyTorch3D triangulated face ids."""
    tri_counts = []
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("f "):
                tri_counts.append(max(len(line.split()) - 3, 0))

    if not tri_counts or sum(tri_counts) != n_tri_faces:
        return face_list[face_list < n_tri_faces]

    if face_list.size and face_list.max() >= len(tri_counts):
        return face_list[face_list < n_tri_faces]

    tri_starts = np.cumsum([0] + tri_counts[:-1])
    remapped = []
    for fid in face_list:
        if 0 <= fid < len(tri_counts):
            start = tri_starts[fid]
            remapped.extend(range(start, start + tri_counts[fid]))
    return np.unique(np.asarray(remapped, dtype=np.int64))


def load_optimizable_faces(faces_path, faces_idx, obj_path=None):
    """Load optimizable face indices. Returns triangulated face ids or None."""
    if not faces_path or faces_path.lower() == "none":
        return None
    if not os.path.exists(faces_path):
        print(f"    Faces file not found: {faces_path}, optimizing full mesh")
        return None
    face_list = np.loadtxt(faces_path, dtype=np.int64)
    face_list = np.atleast_1d(face_list)
    n_faces = faces_idx.shape[0]
    if obj_path is not None:
        original_count = len(face_list)
        face_list = remap_obj_faces_to_triangulated(
            obj_path, face_list, n_faces)
        if len(face_list) != original_count:
            print(
                f"    Remapped {original_count} OBJ faces to {len(face_list)} triangulated faces")
    else:
        face_list = face_list[face_list < n_faces]
    if len(face_list) == 0:
        return None
    return face_list


def cam_trans_to_eye(cam_trans, device):
    loc = cam_trans[0][0:3]
    x, y, z = loc[0], loc[1], loc[2]
    return torch.tensor([x, z, y], dtype=torch.float32, device=device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# Vehicle data container
# ==============================================================================


def load_vehicle(obj_path, face_path, device):
    """Load a single vehicle mesh with precomputed geometry data."""
    verts, faces_idx, verts_uv, faces_uv, texture, texture_path = load_mesh(
        obj_path, device)
    face_list = load_optimizable_faces(face_path, faces_idx, obj_path)
    normals = get_vertex_normals(Meshes(verts=[verts], faces=[faces_idx]))
    bbox_min, bbox_max = compute_mesh_bbox(verts)

    n_opt = len(face_list) if face_list is not None else faces_idx.shape[0]
    print(f"  {Path(obj_path).stem}: {verts.shape[0]} verts, "
          f"{faces_idx.shape[0]} faces, {n_opt} optimizable")
    if texture_path is not None:
        print(f"    texture: {texture_path}")

    return {
        'name': Path(obj_path).stem,
        'verts': verts,
        'faces_idx': faces_idx,
        'verts_uv': verts_uv,
        'faces_uv': faces_uv,
        'texture': texture,
        'texture_path': texture_path,
        'face_list': face_list,
        'n_faces': faces_idx.shape[0],
        'normals': normals,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
    }


# ==============================================================================
# Training
# ==============================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Seed: {args.seed}")

    config = ATFConfig()
    for k, v in vars(args).items():
        if hasattr(config, k) and v is not None:
            setattr(config, k, v)

    lambda_cross = args.lambda_cross
    lambda_local_color_dist = args.lambda_local_color_dist

    # ── YOLOv3 ──
    print("Loading YOLOv3...")
    yolo = load_yolov3(args.weights, args.cfg, args.data, device)
    compute_loss = APAdversarialLoss(temperature=0.5, topk=50)

    # ── Dataset ──
    print("Loading training dataset...")
    npz_dir, label_dir, mask_dir, npz_color_order = resolve_dataset_paths(args)
    items, mask_dir = load_dataset(npz_dir, label_dir, mask_dir)
    print(f"  train: {len(items)} samples from {npz_dir}")
    print(f"  train npz color order: {npz_color_order}")
    if label_dir or mask_dir:
        print(
            f"  train labels: {label_dir or 'rendered vehicle bbox'} | masks: {mask_dir or 'rendered vehicle mask'}")
    else:
        print("  train labels/masks: inferred from rendered vehicle masks")

    # ── Vehicles ──
    n_vehicles = len(args.obj_file)
    faces_paths = args.faces if args.faces else ["none"] * n_vehicles
    if len(faces_paths) < n_vehicles:
        faces_paths = faces_paths * n_vehicles

    print(f"Loading {n_vehicles} vehicle(s)...")
    vehicles = []
    for i in range(n_vehicles):
        v = load_vehicle(args.obj_file[i], faces_paths[i], device)
        vehicles.append(v)

    fallback_gray = torch.tensor([72.0, 72.0, 72.0], device=device) / 255.0
    palette = get_camo_palette(device)
    target_category_dist = get_target_category_dist(device)

    # ── Components ──
    renderer = NvdiffrastRenderer(device, image_size=config.render_size)
    print(f"Renderer: nvdiffrast  |  Vehicles: {n_vehicles}  "
          f"|  Cross-vehicle: {'real' if n_vehicles > 1 else 'simulated (stretch)'}")
    atf = ATFModel(config).to(device)
    print(f"ATF params: {sum(p.numel() for p in atf.parameters()):,}")

    num_samples = len(items)
    if args.epochs is not None:
        config.total_iters = args.epochs * num_samples
    iters_per_epoch = num_samples

    optimizer = torch.optim.AdamW([
        {"params": atf.encoder.parameters(), "lr": config.lr_encoder},
        {"params": atf.decoder.parameters(), "lr": config.lr_decoder},
    ], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.total_iters)

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    img_dir = out_dir / "images"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    save_palette_visualization(
        palette, target_category_dist, str(img_dir / "woodland_palette.png"))
    save_interval = getattr(args, "save_interval", 500)
    save_training_params(args, config, vehicles, faces_paths, num_samples,
                         iters_per_epoch, device, out_dir)

    # ── Training loop ──
    pbar = tqdm(range(config.total_iters), desc="ATF")
    loss_log = []          # list of {key: scalar}, one dict per step
    epoch_losses = []      # (iteration, {key: avg}) per epoch
    running = {}            # running sum per component

    for it in pbar:
        # ── Select vehicle (uniform) ──
        vi = torch.randint(0, n_vehicles, (1,)).item()
        v = vehicles[vi]

        # ── Load data (deterministic order) ──
        npz_path, label_path = items[it % len(items)]
        background, mask, targets, cam_trans = load_npz_data(
            npz_path, label_path, mask_dir, config.render_size, device,
            inferred_class=args.target_class if args.target_class >= 0 else 2,
            color_order=npz_color_order)

        # ── Geometric augmentation ──
        if torch.rand(1).item() < config.geo_aug_prob:
            aug_v = stretch_mesh(v['verts'], config.geo_aug_scale)
        else:
            aug_v = v['verts']

        aug_mesh = Meshes(verts=[aug_v], faces=[v['faces_idx']])
        aug_normals = get_vertex_normals(aug_mesh)
        aug_bbox_min, aug_bbox_max = compute_mesh_bbox(aug_v)

        eye = cam_trans_to_eye(cam_trans, device)

        # ── Rasterize + per-pixel ATF query ──
        surf_pts, surf_norms, pixel_idx, veh_mask, face_ids, pixel_uv = \
            renderer.rasterize_surface_points(
                aug_v, v['faces_idx'], eye, v['verts_uv'], v['faces_uv'])
        n_query = surf_pts.shape[0] if surf_pts is not None else 0
        if mask is None:
            mask = veh_mask.float().squeeze(0)
        if targets is None:
            targets = mask_to_target(
                mask, device,
                class_id=args.target_class if args.target_class >= 0 else 2)

        if n_query > 0:
            surf_pts_all, surf_norms_all = surf_pts, surf_norms
            query_01_all = normalize_points_with_margin(
                surf_pts_all, aug_bbox_min, aug_bbox_max, margin=config.bbox_margin)
            geo_feats_all = extract_geometric_features(
                aug_v, v['faces_idx'], aug_normals, surf_pts_all, surf_norms_all,
                aug_bbox_min, aug_bbox_max)

            infer_bs = config.num_surface_points
            # Voxel quantization: points in same cell share one ATF query.
            voxel_idx, voxel_inv, n_voxels, uniq_grid = voxelize_surface_points(
                surf_pts_all, aug_bbox_min, aug_bbox_max, config.block_resolution)

            alphas_vox = torch.zeros(n_voxels, 1, device=device)
            cat_logits_vox = torch.zeros(
                n_voxels, palette.shape[0], device=device)
            sub_logits_vox = torch.zeros(
                n_voxels, palette.shape[0], palette.shape[1], device=device)
            for start in range(0, n_voxels, infer_bs):
                end = min(start + infer_bs, n_voxels)
                atf.encoder.reset_planes()
                a_batch, cat_batch, sub_batch = atf(
                    query_01_all[voxel_idx[start:end]],
                    geo_feats_all[voxel_idx[start:end]])
                alphas_vox[start:end] = a_batch
                cat_logits_vox[start:end] = cat_batch
                sub_logits_vox[start:end] = sub_batch

            pixel_colors, category_probs_vox, _ = compose_palette_camo(
                alphas_vox, cat_logits_vox, sub_logits_vox, uniq_grid, voxel_inv,
                palette, config.category_temp, config.subcolor_temp)

            # Face mask (vehicle-specific): keep original map_Kd texture outside optimized faces.
            if v['face_list'] is not None and face_ids is not None:
                pixel_colors = pixel_colors.clone()
                opt_mask = torch.zeros(
                    v['n_faces'], dtype=torch.bool, device=device)
                opt_mask[torch.tensor(v['face_list'], device=device)] = True
                non_opt = ~opt_mask[face_ids]
                original_colors = sample_texture(v['texture'], pixel_uv)
                if original_colors is None:
                    original_colors = fallback_gray.expand(
                        pixel_colors.shape[0], 3)
                pixel_colors[non_opt] = original_colors[non_opt]

            rendered = renderer.render_from_pixel_colors(
                pixel_colors, pixel_idx, veh_mask)
            composite = renderer.render_from_pixel_colors(
                pixel_colors, pixel_idx, veh_mask, background=background)
        else:
            rendered = background
            composite = background

        query_pts, query_norms = surf_pts, surf_norms

        # ── YOLO forward ──
        yolo_in = composite.unsqueeze(0)
        yolo_out = yolo(yolo_in)
        yolo_pred = yolo_out[0] if isinstance(yolo_out, tuple) else yolo_out
        adv_loss = compute_loss(
            yolo_pred, targets, target_class=None if args.target_class < 0 else args.target_class)

        # ── Cross-vehicle consistency ──
        L_cross = torch.tensor(0.0, device=device)
        if lambda_cross > 0 and n_query > 0:
            if n_vehicles > 1:
                # Real cross-vehicle: sample from a different vehicle
                vj = vi
                while vj == vi:
                    vj = torch.randint(0, n_vehicles, (1,)).item()
                v2 = vehicles[vj]

                n2 = min(config.num_surface_points, v2['verts'].shape[0])
                idx2 = torch.randperm(v2['verts'].shape[0], device=device)[:n2]
                q2_pts = v2['verts'][idx2]
                q2_norms = v2['normals'][idx2]

                q2_01 = normalize_points_with_margin(
                    q2_pts, v2['bbox_min'], v2['bbox_max'],
                    margin=config.bbox_margin)
                geo_feats2 = extract_geometric_features(
                    v2['verts'], v2['faces_idx'], v2['normals'],
                    q2_pts, q2_norms, v2['bbox_min'], v2['bbox_max'])
            else:
                # Simulated: stretch same vehicle as pseudo-second-vehicle
                aug_v2 = stretch_mesh(v['verts'], (0.80, 1.20))
                aug_normals2 = get_vertex_normals(
                    Meshes(verts=[aug_v2], faces=[v['faces_idx']]))
                aug_bb_min2, aug_bb_max2 = compute_mesh_bbox(aug_v2)
                q2_pts, q2_norms = aug_v2, aug_normals2

                q2_01 = normalize_points_with_margin(
                    q2_pts, aug_bb_min2, aug_bb_max2, margin=config.bbox_margin)
                geo_feats2 = extract_geometric_features(
                    aug_v2, v['faces_idx'], aug_normals2,
                    q2_pts, q2_norms, aug_bb_min2, aug_bb_max2)

            cross_idx1 = torch.randperm(n_query, device=device)[
                :min(config.num_surface_points, n_query)]
            cross_idx2 = torch.randperm(q2_pts.shape[0], device=device)[
                :min(config.num_surface_points, q2_pts.shape[0])]
            L_cross = cross_vehicle_consistency_loss(
                atf, [(query_01_all[cross_idx1], geo_feats_all[cross_idx1]),
                      (q2_01[cross_idx2], geo_feats2[cross_idx2])])

        # ── Local color-distribution regularization ──
        if n_query > 0:
            L_local_color_dist = local_category_distribution_loss(
                cat_logits_vox, target_category_dist, uniq_grid,
                config.block_resolution, config.local_color_dist_resolution,
                alphas=alphas_vox, temperature=config.category_temp)
        else:
            L_local_color_dist = torch.tensor(0.0, device=device)

        # ── Total loss ──
        total_loss = adv_loss + lambda_cross * L_cross \
            + lambda_local_color_dist * L_local_color_dist

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(atf.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        adv_val, cross_val = adv_loss.item(), L_cross.item()
        local_color_dist_val = L_local_color_dist.item()
        total_val = adv_val + lambda_cross * cross_val \
            + lambda_local_color_dist * local_color_dist_val

        step_losses = {'adv': adv_val, 'cross': cross_val,
                       'local_color_dist': local_color_dist_val,
                       'total': total_val}
        loss_log.append(step_losses)
        for loss_name, loss_value in step_losses.items():
            running[loss_name] = running.get(loss_name, 0.0) + loss_value

        epoch = (it + 1) // iters_per_epoch
        if (it + 1) % iters_per_epoch == 0:
            epoch_avg = {k: value / iters_per_epoch for k,
                         value in running.items()}
            epoch_losses.append((it + 1, epoch_avg))
            running = {}
            print(f"\nEpoch {epoch}  "
                  f"adv={epoch_avg['adv']:.2f} "
                  f"cross={epoch_avg['cross']:.2f} "
                  f"lcdist={epoch_avg['local_color_dist']:.2f} "
                  f"total={epoch_avg['total']:.2f}")

            epoch_ckpt_path = ckpt_dir / f"atf_epoch_{epoch:03d}.pt"
            save_checkpoint(atf, optimizer, it + 1, config,
                            str(epoch_ckpt_path), epoch=epoch)
            print(f"Epoch checkpoint: {epoch_ckpt_path}")

        display_epoch = (it // iters_per_epoch) + 1
        pbar.set_postfix(
            adv=f"{adv_val:.2f}", cross=f"{cross_val:.2f}",
            lcdist=f"{local_color_dist_val:.3f}",
            car=v['name'], ep=f"{display_epoch}",
        )

        # ── Save debug images + loss curve ──
        should_save_preview = (it == 0 or it + 1 == config.total_iters or
                               (save_interval and (it + 1) % save_interval == 0))
        if should_save_preview:
            npz_stem = Path(npz_path).stem
            save_image_with_boxes(composite, targets, str(
                img_dir / f"composite_box_{it+1:05d}_{v['name']}_{npz_stem}.png"))
            with open(out_dir / "loss_history.json", "w") as f:
                json.dump(loss_log, f, indent=2)
            save_loss_curve(loss_log, epoch_losses,
                            str(img_dir / f"loss_curve_{it+1:05d}.png"))

        atf.encoder.reset_planes()

    final_ckpt_path = str(ckpt_dir / "atf_final.pt")
    final_epoch = (config.total_iters + iters_per_epoch - 1) // iters_per_epoch
    save_checkpoint(atf, optimizer, config.total_iters, config, final_ckpt_path,
                    epoch=final_epoch)
    print(f"Final checkpoint: {final_ckpt_path}")
    print(f"\nTraining done. Output: {out_dir}")

    with open(out_dir / "epoch_losses.json", "w") as f:
        json.dump([(it, avg) for it, avg in epoch_losses], f, indent=2)


def save_loss_curve(loss_log, epoch_losses, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Extract per-component series from dict-based loss_log
    components = ['total', 'adv', 'cross', 'local_color_dist']
    colors = {'total': '#d62728', 'adv': '#1f77b4',
              'cross': '#9467bd', 'local_color_dist': '#bcbd22'}
    series = {k: [d[k] for d in loss_log] for k in components}

    fig, ax = plt.subplots(figsize=(12, 5))

    window = max(1, len(loss_log) // 200)
    for name in components:
        vals = series[name]
        if window > 1:
            smoothed = np.convolve(vals, np.ones(
                window) / window, mode='valid')
            ax.plot(range(window - 1, len(vals)), smoothed, linewidth=1.0,
                    color=colors[name], label=f'{name} (w={window})')
        else:
            ax.plot(vals, alpha=0.3, linewidth=0.5, color=colors[name],
                    label=f'{name} (raw)')

    if epoch_losses:
        ep_iters, ep_avgs = zip(*epoch_losses)
        ep_total = [a['total'] for a in ep_avgs]
        ax.scatter(ep_iters, ep_total, marker='o', s=30, color='#d62728',
                   zorder=5, label='epoch total')

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("ATF Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train the GeoCamo surface field")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to a pretrained YOLOv3 checkpoint")
    p.add_argument("--cfg", type=str, default="")
    p.add_argument("--data", type=str, default="data/train.yaml")
    p.add_argument("--obj-file", type=str, nargs='+',
                   required=True,
                   help="Training vehicle OBJ file(s). Multiple = multi-vehicle joint training")
    p.add_argument("--npz-dir", type=str, default=None,
                   help="Directory containing training NPZ files; overrides data yaml")
    p.add_argument("--label-dir", type=str, default=None,
                   help="Optional YOLO label directory; inferred from rendered vehicle mask if omitted")
    p.add_argument("--mask-dir", type=str, default=None,
                   help="Optional mask directory; inferred from rendered vehicle mask if omitted")
    p.add_argument("--npz-color-order", choices=["rgb", "bgr"], default=None,
                   help="Color order of img stored in NPZ; overrides data yaml")
    p.add_argument("--output-dir", type=str, default="results/atf_training")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--num-surface-points", type=int, default=None)
    p.add_argument("--lr-encoder", type=float, default=None)
    p.add_argument("--lr-decoder", type=float, default=None)
    p.add_argument("--render-size", type=int, default=None)
    p.add_argument("--block-resolution", type=int, default=None)
    p.add_argument("--category-temp", type=float, default=None,
                   help="Softmax temperature for palette categories")
    p.add_argument("--subcolor-temp", type=float, default=None,
                   help="Softmax temperature for category subcolors")
    p.add_argument("--local-color-dist-resolution", type=int, default=None,
                   help="Coarse 3D grid resolution for local color distribution loss")
    p.add_argument("--save-interval", type=int, default=500,
                   help="Interval for saving debug images and loss curves")
    p.add_argument("--faces", type=str, nargs='*',
                   default=None,
                   help="Face list file(s) per vehicle (none = full mesh)")
    p.add_argument("--no-learned-geo-key", dest="use_learned_geo_key",
                   action="store_false", default=True,
                   help="Disable GeoFeatureRouter, use hand-picked geo key")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-cross", type=float, default=0.02,
                   help="Cross-vehicle consistency weight (real vehicles when >1)")
    p.add_argument("--lambda-local-color-dist", type=float, default=0.1,
                   help="Match category distribution inside local spatial regions")
    p.add_argument("--target-class", type=int, default=2,
                   help="Detector class id to suppress; use -1 to read class from labels")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
