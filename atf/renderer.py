import math
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


class NvdiffrastRenderer:
    """nvdiffrast-based renderer — direct rasterization without lighting.

    Unlike PyTorch3D's SoftPhongShader which adds ambient/diffuse/specular lighting,
    this renderer directly rasterizes vertex colors. This gives:
      - Cleaner gradients (no lighting layer between ATF output and rendered image)
      - Faster rendering (CUDA-accelerated rasterization)
      - Exact color control (ATF output = rendered pixel color after interpolation)

    Uses OpenGL-compatible projection matrix (not PyTorch3D's NDC convention).
    """

    def __init__(self, device, image_size=608):
        self.device = device
        self.image_size = image_size
        self.glctx = dr.RasterizeCudaContext()

    def build_mesh(self, verts, faces, verts_colors):
        """Store mesh data. Returns dict for later rasterization."""
        return {
            "verts": verts.to(self.device),
            "faces": faces.int().to(self.device),
            "verts_colors": verts_colors.to(self.device),
        }

    @staticmethod
    def _build_opengl_mvp(eye, at=(0., 0., 0.), up=(0., 1., 0.),
                           fov=90., aspect=1.0, near=0.1, far=100.):
        """Build OpenGL-compatible Model-View-Projection matrix.

        clip = P @ V @ world_point

        Args:
            eye: (3,) camera position (Y-up convention)
            at: (3,) look-at target
            up: (3,) up vector
            fov: field of view in degrees
            aspect: width / height
            near, far: near and far clipping planes

        Returns:
            mvp: (4, 4) combined MVP matrix
        """
        device = eye.device
        dtype = eye.dtype

        # --- View matrix: lookAt(eye, at, up) ---
        z_axis = F.normalize(eye - torch.tensor(at, device=device, dtype=dtype), dim=-1)
        up_vec = torch.tensor(up, device=device, dtype=dtype).expand_as(z_axis)

        # Handle degenerate case: up parallel to view direction
        cross_up_z = torch.cross(up_vec, z_axis)
        if cross_up_z.norm() < 1e-6:
            up_vec = torch.tensor([0., 0., -1.], device=device, dtype=dtype).expand_as(z_axis)
            cross_up_z = torch.cross(up_vec, z_axis)

        x_axis = F.normalize(cross_up_z, dim=-1)
        y_axis = torch.cross(z_axis, x_axis)

        R_v = torch.stack([x_axis, y_axis, z_axis], dim=0)  # (3, 3) world→cam rotation
        t_v = -R_v @ eye  # (3,) translation

        V = torch.eye(4, device=device, dtype=dtype)
        V[:3, :3] = R_v
        V[:3, 3] = t_v

        # --- Projection matrix (OpenGL perspective) ---
        fov_rad = fov * (math.pi / 180.0)
        f = 1.0 / math.tan(fov_rad / 2.0)

        P = torch.zeros(4, 4, device=device, dtype=dtype)
        P[0, 0] = f / aspect
        P[1, 1] = f
        P[2, 2] = (far + near) / (near - far)
        P[2, 3] = (2.0 * far * near) / (near - far)
        P[3, 2] = -1.0

        return P @ V

    def render_with_mvp(self, mesh_data, mvp, h=None, w=None):
        """Render mesh using model-view-projection matrix.

        Args:
            mesh_data: dict from build_mesh()
            mvp: (4, 4) OpenGL-style model-view-projection matrix
            h, w: output resolution (default: self.image_size)

        Returns:
            image: (3, H, W) rendered image in [0, 1]
        """
        if h is None:
            h = self.image_size
        if w is None:
            w = self.image_size

        verts = mesh_data["verts"]
        faces = mesh_data["faces"]
        colors = mesh_data["verts_colors"]

        # Transform vertices to clip space: clip = mvp @ world (column vector)
        verts_homo = F.pad(verts, (0, 1), value=1.0)  # (N, 4)
        verts_clip = ((mvp @ verts_homo.T).T).float()  # (N, 4)
        verts_clip = verts_clip.unsqueeze(0).contiguous()  # (1, N, 4) — must be contiguous

        # Rasterize
        rast, _ = dr.rasterize(self.glctx, verts_clip, faces, (h, w))

        # Interpolate vertex colors (must be contiguous)
        colors_batch = colors.unsqueeze(0).contiguous()  # (1, N, 3)
        colors_interp, _ = dr.interpolate(colors_batch, rast, faces)

        # Anti-aliasing (all inputs must be contiguous)
        colors_aa = dr.antialias(colors_interp.contiguous(), rast, verts_clip, faces)

        image = colors_aa.squeeze(0).clamp(0.0, 1.0)  # (h, w, 3)
        # Flip Y: nvdiffrast/OpenGL origin is bottom-left, image convention is top-left
        image = torch.flip(image, dims=[0])
        image = image.permute(2, 0, 1).contiguous()  # (3, h, w)
        return image

    def render_face_colors_with_mvp(self, mesh_data, face_colors, mvp, h=None, w=None):
        """Render using per-face colors (no barycentric interpolation).

        Each triangle gets a single color. Adjacent triangles can have
        sharply different colors — better for adversarial high-frequency patterns.

        Args:
            mesh_data: dict from build_mesh() with "verts" and "faces"
            face_colors: (F, 3) per-face colors in [0, 1]
            mvp: (4, 4) OpenGL MVP matrix
            h, w: output resolution

        Returns:
            image: (3, H, W) rendered image in [0, 1]
        """
        if h is None:
            h = self.image_size
        if w is None:
            w = self.image_size

        verts = mesh_data["verts"]
        faces = mesh_data["faces"]

        # Transform vertices to clip space
        verts_homo = F.pad(verts, (0, 1), value=1.0)
        verts_clip = ((mvp @ verts_homo.T).T).float().unsqueeze(0).contiguous()

        # Rasterize — get triangle_id per pixel (rast[...,-1] is 1-indexed tri_id)
        rast, _ = dr.rasterize(self.glctx, verts_clip, faces, (h, w))

        # Direct face color lookup (no interpolation)
        tri_id = rast[0, ..., -1].long() - 1  # (H, W), 0-indexed, -1=background
        mask = (tri_id >= 0).unsqueeze(-1)  # (H, W, 1)

        safe_id = tri_id.clamp(0)
        image = face_colors[safe_id]  # (H, W, 3)
        image = image * mask.float()  # zero background
        image = image.clamp(0.0, 1.0)

        # Flip Y: OpenGL origin bottom-left → image origin top-left
        image = torch.flip(image, dims=[0])
        image = image.permute(2, 0, 1).contiguous()  # (3, H, W)
        return image

    def bake_texture(self, verts_uv, faces_uv, colors, tex_size=1024,
                      color_mode="vertex"):
        """Bake ATF colors into a UV texture image.

        Renders at 2× resolution then downsamples for antialiasing,
        ensuring sub-pixel triangles are captured.

        Args:
            verts_uv: (N, 2) UV coordinates in [0, 1]²
            faces_uv: (F, 3) UV face indices
            colors: (N, 3) vertex colors OR (F, 3) face colors
            tex_size: texture resolution (square)
            color_mode: "vertex" (interpolated) or "face" (flat per-face)

        Returns:
            texture: (3, tex_size, tex_size) image in [0, 1]
        """
        # Render at 2× resolution for sub-pixel face coverage, then downsample
        render_size = tex_size * 2

        # Transform UV coords from [0,1] to OpenGL clip space [-1,1]
        uv_clip = torch.cat([
            verts_uv * 2.0 - 1.0,
            torch.zeros(verts_uv.shape[0], 1, device=verts_uv.device),
            torch.ones(verts_uv.shape[0], 1, device=verts_uv.device),
        ], dim=-1).float().unsqueeze(0).contiguous()  # (1, N, 4)

        rast, _ = dr.rasterize(self.glctx, uv_clip, faces_uv.int(),
                               (render_size, render_size))

        if color_mode == "face":
            tri_id = rast[0, ..., -1].long() - 1
            mask = (tri_id >= 0).unsqueeze(-1)
            tex = colors[tri_id.clamp(0)] * mask.float()
        else:
            colors_batch = colors.unsqueeze(0).contiguous()
            tex_interp, _ = dr.interpolate(colors_batch, rast, faces_uv.int())
            tex = tex_interp.squeeze(0)

        tex = tex.clamp(0.0, 1.0)
        tex = torch.flip(tex, dims=[0])
        tex = tex.permute(2, 0, 1).contiguous()  # (3, H, W)

        # Downsample to target resolution
        tex = tex.unsqueeze(0)  # (1, 3, H, W)
        tex = F.interpolate(tex, size=(tex_size, tex_size), mode='bilinear',
                            align_corners=False)
        tex = tex.squeeze(0).contiguous()
        return tex

    def bake_uv_texture_from_atf(self, verts_3d, faces_3d, verts_uv, faces_uv,
                                   geo_features_fn, atf_model, tex_size=1024,
                                   face_list=None, block_resolution=32,
                                   bbox_margin=0.2, category_temp=1.0,
                                   subcolor_temp=1.0, original_texture=None):
        """Bake UV texture by per-texel ATF query (for UV color mode).

        Unlike bake_texture which uses pre-computed per-face colors, this
        rasterizes in UV space, maps each texel to a 3D surface point,
        and queries ATF independently per texel. This captures intra-face
        color variation that UV training mode learns.

        Args:
            verts_3d: (N, 3) 3D vertex positions
            faces_3d: (F, 3) 3D face indices
            verts_uv: (U, 2) UV coordinates
            faces_uv: (F, 3) UV face indices (same topology as faces_3d)
            geo_features_fn: callable(pts, norms) → (M, 11) geometric features
            atf_model: ATFModel for per-point color query
            tex_size: output texture resolution
            face_list: optional list of optimizable face indices; non-optimized
                       faces keep original_texture when available
            original_texture: optional (3, H, W) source map_Kd texture

        Returns:
            texture: (3, tex_size, tex_size) image in [0, 1]
        """
        device = verts_3d.device
        render_size = tex_size * 2
        batch_size = 4096

        # Rasterize UV mesh
        uv_clip = torch.cat([
            verts_uv * 2.0 - 1.0,
            torch.zeros(verts_uv.shape[0], 1, device=device),
            torch.ones(verts_uv.shape[0], 1, device=device),
        ], dim=-1).float().unsqueeze(0).contiguous()

        rast, _ = dr.rasterize(self.glctx, uv_clip, faces_uv.int(),
                               (render_size, render_size))

        # Interpolate 3D positions at each UV texel (same face topology)
        pts3d_batch = verts_3d.unsqueeze(0).contiguous()
        surf_pts, _ = dr.interpolate(pts3d_batch, rast.contiguous(),
                                      faces_3d.int())  # (1, H, W, 3)

        # Interpolate normals
        from pytorch3d.structures import Meshes
        from atf.utils import get_vertex_normals
        mesh_p3d = Meshes(verts=[verts_3d], faces=[faces_3d.int()])
        vnorms = get_vertex_normals(mesh_p3d)
        norms_batch = vnorms.unsqueeze(0).contiguous()
        surf_norms, _ = dr.interpolate(norms_batch, rast.contiguous(),
                                        faces_3d.int())
        surf_norms = F.normalize(surf_norms, dim=-1)

        # Identify covered texels
        mask = (rast[0, ..., -1] > 0)  # (H, W), vehicle texels
        rows, cols = torch.where(mask)

        if rows.shape[0] == 0:
            return torch.zeros(3, tex_size, tex_size, device=device)

        # Gather 3D points and normals for covered texels
        surf_pts_flat = surf_pts[0, rows, cols]  # (M, 3)
        surf_norms_flat = surf_norms[0, rows, cols]  # (M, 3)
        M = rows.shape[0]

        # Voxel quantization for digital camouflage block pattern
        from atf.utils import voxelize_surface_points
        from atf.camo import get_camo_palette, compose_palette_camo
        bbox_min = verts_3d.min(dim=0).values
        bbox_max = verts_3d.max(dim=0).values
        block_res = block_resolution

        # Query ATF colors.
        from atf.utils import normalize_points_with_margin
        voxel_idx, voxel_inv, n_voxels, texel_grid = voxelize_surface_points(
            surf_pts_flat, bbox_min, bbox_max, block_res)

        palette = get_camo_palette(device)
        alphas_vox = torch.zeros(n_voxels, 1, device=device)
        cat_logits_vox = torch.zeros(n_voxels, palette.shape[0], device=device)
        sub_logits_vox = torch.zeros(
            n_voxels, palette.shape[0], palette.shape[1], device=device)
        for start in range(0, n_voxels, batch_size):
            end = min(start + batch_size, n_voxels)
            pts_batch = surf_pts_flat[voxel_idx[start:end]].contiguous()
            norms_batch = surf_norms_flat[voxel_idx[start:end]].contiguous()
            geo_batch = geo_features_fn(pts_batch, norms_batch)
            atf_model.encoder.reset_planes()
            query_01 = normalize_points_with_margin(
                pts_batch, bbox_min, bbox_max, margin=bbox_margin)
            a_batch, cat_batch, sub_batch = atf_model(query_01, geo_batch)
            alphas_vox[start:end] = a_batch
            cat_logits_vox[start:end] = cat_batch
            sub_logits_vox[start:end] = sub_batch

        texel_colors, _, _ = compose_palette_camo(
            alphas_vox, cat_logits_vox, sub_logits_vox, texel_grid, voxel_inv,
            palette, category_temp, subcolor_temp)

        if face_list is not None:
            rast_fids = (rast[0, ..., -1].long() - 1)  # 1-indexed -> 0-indexed
            texel_fids = rast_fids[rows, cols]
            opt_mask = torch.zeros(faces_3d.shape[0], dtype=torch.bool, device=device)
            opt_mask[torch.tensor(face_list, device=device)] = True
            non_opt = ~opt_mask[texel_fids]
            if original_texture is not None:
                uv_attr = verts_uv.unsqueeze(0).contiguous()
                uv_interp, _ = dr.interpolate(uv_attr, rast.contiguous(), faces_uv.int())
                texel_uv = uv_interp[0, rows, cols].clamp(0.0, 1.0)
                grid = torch.empty(1, texel_uv.shape[0], 1, 2, device=device,
                                   dtype=original_texture.dtype)
                grid[0, :, 0, 0] = texel_uv[:, 0] * 2.0 - 1.0
                grid[0, :, 0, 1] = 1.0 - texel_uv[:, 1] * 2.0
                original_colors = F.grid_sample(
                    original_texture.unsqueeze(0),
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                ).squeeze(0).squeeze(-1).T.contiguous()
            else:
                gray = torch.tensor([72.0 / 255.0] * 3, device=device)
                original_colors = gray.expand(texel_colors.shape[0], 3)
            texel_colors[non_opt] = original_colors[non_opt]

        # Build texture image
        tex = torch.zeros(render_size, render_size, 3, device=device)
        tex[rows, cols] = texel_colors

        tex = tex.clamp(0.0, 1.0)
        tex = torch.flip(tex, dims=[0])  # OpenGL → image convention
        tex = tex.permute(2, 0, 1).contiguous()  # (3, H, W)

        # Downsample
        tex = tex.unsqueeze(0)
        tex = F.interpolate(tex, size=(tex_size, tex_size), mode='bilinear',
                            align_corners=False)
        tex = tex.squeeze(0).contiguous()
        return tex

    def rasterize_surface_points(self, verts, faces, eye, verts_uv=None, faces_uv=None,
                                 fov=90., h=None, w=None):
        """Rasterize mesh and return per-pixel 3D surface data for ATF queries.

        Maps each screen pixel hitting the vehicle to its 3D position and normal
        via barycentric interpolation. Used by UV color mode for per-pixel ATF.

        Args:
            verts: (N, 3) vertex positions
            faces: (F, 3) face indices
            eye: (3,) camera position
            verts_uv: optional (U, 2) UV coordinates for texture lookup
            faces_uv: optional (F, 3) UV indices for texture lookup
            fov: field of view in degrees
            h, w: output resolution

        Returns:
            surf_pts: (M, 3) 3D positions at vehicle pixels (None if no vehicle visible)
            surf_norms: (M, 3) normals at vehicle pixels
            pixel_idx: (M, 2) (row, col) indices of vehicle pixels
            mask: (1, H, W) boolean mask
            face_ids: (M,) face index per pixel (0-indexed, for face masking)
            pixel_uv: (M, 2) interpolated UV coordinates, or None
        """
        if h is None:
            h = self.image_size
        if w is None:
            w = self.image_size

        mesh_data = self.build_mesh(verts, faces,
                                     torch.zeros(verts.shape[0], 3, device=self.device))
        mvp = self._build_opengl_mvp(eye=eye, fov=fov, aspect=w / h)

        verts_homo = F.pad(mesh_data["verts"], (0, 1), value=1.0)
        verts_clip = ((mvp @ verts_homo.T).T).float().unsqueeze(0).contiguous()

        # Rasterize to get per-pixel face_id and barycentrics
        rast, _ = dr.rasterize(self.glctx, verts_clip, faces.int(), (h, w))

        # Interpolate 3D positions and normals at each pixel
        rast_contig = rast.contiguous()
        pts_batch = verts.unsqueeze(0).contiguous()
        surf_pts, _ = dr.interpolate(pts_batch, rast_contig, faces.int())  # (1, H, W, 3)

        # Compute vertex normals and interpolate
        from atf.utils import get_vertex_normals
        from pytorch3d.structures import Meshes
        mesh_p3d = Meshes(verts=[verts], faces=[faces.int()])
        vnorms = get_vertex_normals(mesh_p3d)
        norms_batch = vnorms.unsqueeze(0).contiguous()
        surf_norms_interp, _ = dr.interpolate(norms_batch, rast_contig, faces.int())
        surf_norms_interp = F.normalize(surf_norms_interp, dim=-1)

        uv_interp = None
        if verts_uv is not None and faces_uv is not None:
            uv_batch = verts_uv.unsqueeze(0).contiguous()
            uv_interp, _ = dr.interpolate(uv_batch, rast_contig, faces_uv.int())

        # Extract vehicle pixels
        mask = (rast[0, ..., -1] > 0).float()  # (H, W)
        # Flip Y: OpenGL bottom-left → image top-left
        mask = torch.flip(mask, dims=[0])
        veh_mask = mask > 0.5

        # Build per-pixel data for vehicle region
        # surf_pts is in OpenGL convention (bottom-left), flip Y
        surf_pts = torch.flip(surf_pts.squeeze(0), dims=[0])  # (H, W, 3)
        surf_norms_interp = torch.flip(surf_norms_interp.squeeze(0), dims=[0])  # (H, W, 3)
        tri_id = rast[0, ..., -1].long() - 1  # (H, W), 0-indexed face id
        tri_id = torch.flip(tri_id, dims=[0])  # flip Y to match image convention
        if uv_interp is not None:
            uv_interp = torch.flip(uv_interp.squeeze(0), dims=[0]).clamp(0.0, 1.0)

        rows, cols = torch.where(veh_mask)
        if rows.shape[0] == 0:
            return None, None, None, veh_mask.unsqueeze(0), None, None

        surf_pts_flat = surf_pts[rows, cols]  # (M, 3)
        surf_norms_flat = surf_norms_interp[rows, cols]  # (M, 3)
        pixel_idx = torch.stack([rows, cols], dim=-1)  # (M, 2)
        face_ids = tri_id[rows, cols]  # (M,)
        pixel_uv = uv_interp[rows, cols] if uv_interp is not None else None

        return surf_pts_flat.contiguous(), surf_norms_flat.contiguous(), \
            pixel_idx, veh_mask.unsqueeze(0), face_ids, pixel_uv

    def render_from_pixel_colors(self, pixel_colors, pixel_idx, mask,
                                  h=None, w=None, background=None):
        """Build image from per-pixel colors.

        Args:
            pixel_colors: (M, 3) colors in [0, 1] for each vehicle pixel
            pixel_idx: (M, 2) (row, col) indices
            mask: (1, H, W) vehicle mask
            h, w: resolution
            background: optional (3, H, W) background image

        Returns:
            image: (3, H, W) rendered image
        """
        if h is None:
            h = self.image_size
        if w is None:
            w = self.image_size

        image = torch.zeros(3, h, w, device=pixel_colors.device)
        image[:, pixel_idx[:, 0], pixel_idx[:, 1]] = pixel_colors.T

        if background is not None:
            mask_f = mask.float()
            image = mask_f * image + (1 - mask_f) * background

        return image

    def render_face_colors_from_eye(self, verts, faces, face_colors,
                                     eye, fov=90.):
        """Render per-face colors with camera at 'eye'.

        Args:
            verts: (N, 3), faces: (F, 3), face_colors: (F, 3) in [0, 1]
            eye: (3,) camera position, fov: field of view in degrees
        """
        mesh_data = self.build_mesh(verts, faces,
                                     torch.zeros(verts.shape[0], 3, device=self.device))
        h, w = self.image_size, self.image_size
        mvp = self._build_opengl_mvp(eye=eye, fov=fov, aspect=w / h)
        return self.render_face_colors_with_mvp(mesh_data, face_colors, mvp, h, w)

    def render_from_eye(self, verts, faces, verts_colors, eye, fov=90.):
        """Render with camera at 'eye' looking at origin.

        Args:
            verts: (N, 3) vertex positions
            faces: (F, 3) face indices
            verts_colors: (N, 3) vertex colors in [0, 1]
            eye: (3,) camera position in world space (Y-up)
            fov: field of view in degrees

        Returns:
            image: (3, H, W) rendered image in [0, 1]
        """
        mesh_data = self.build_mesh(verts, faces, verts_colors)
        h, w = self.image_size, self.image_size
        mvp = self._build_opengl_mvp(eye=eye, fov=fov, aspect=w / h)
        return self.render_with_mvp(mesh_data, mvp)
