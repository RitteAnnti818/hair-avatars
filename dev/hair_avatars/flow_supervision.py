"""
A-2: optical-flow cross-frame consistency loss.

Given the model's hair Gaussian positions at adjacent timesteps t and t+1 (same camera), project both
into that camera's pixel space and compare the resulting 2D displacement to real optical flow measured
on the GT video (RAFT, cached lazily per (subject, camera, t) the first time it's needed).

Does not touch train.py's main shuffle=True single-frame loop -- this is meant to be called as a
low-frequency auxiliary step (e.g. every N iterations), on top of the existing photometric training.
"""
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

CACHE_DIR = "/home/hee/hee_data/GaussianAvatars/dev/hair_avatars/flow_cache"


def project_points_to_pixels(xyz_world, camera):
    """xyz_world: (N, 3). Returns (N, 2) pixel coordinates, using the same row-vector
    world_view_transform / full_proj_transform convention as scene/cameras.py."""
    N = xyz_world.shape[0]
    ones = xyz_world.new_ones(N, 1)
    p_h = torch.cat([xyz_world, ones], dim=1)  # (N, 4)
    p_clip = p_h @ camera.full_proj_transform.to(xyz_world.device).to(xyz_world.dtype)  # (N, 4)
    ndc = p_clip[:, :3] / p_clip[:, 3:4].clamp(min=1e-6)
    px = (ndc[:, 0] + 1.0) * 0.5 * camera.image_width
    py = (ndc[:, 1] + 1.0) * 0.5 * camera.image_height
    return torch.stack([px, py], dim=1)  # (N, 2)


class FlowCache:
    """Lazy RAFT flow computation, cached to disk as {subject}/{camera:02d}_{t:05d}.npy (flow from
    t to t+1). Also stores a forward-backward consistency map for optional confidence weighting."""

    def __init__(self, subject, gt_dir, device="cuda"):
        self.subject = subject
        self.gt_dir = Path(gt_dir)  # e.g. output/free60k_baseline_{subj}/train/ours_60000/gt-equivalent;
        # for training frames we don't have pre-rendered GT crops the way test does, so this reads
        # straight from the dataset's own images/ directory instead (see get() below).
        self.device = device
        self._raft = None
        self._transforms = None
        self._mem_cache = {}
        Path(CACHE_DIR, subject).mkdir(parents=True, exist_ok=True)

    def _ensure_raft(self):
        if self._raft is None:
            weights = Raft_Large_Weights.DEFAULT
            self._raft = raft_large(weights=weights).to(self.device).eval()
            self._transforms = weights.transforms()

    @staticmethod
    def _load_img(path):
        from PIL import Image
        return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).contiguous()

    def get(self, img_path_a, img_path_b, cam_idx, t):
        """Returns (flow (2,H,W) float32 tensor, fb_confidence (H,W) float32 tensor), both on CPU.
        flow[:, y, x] is the displacement (in pixels, dx, dy) of the point at pixel (x, y) in frame a,
        as it appears in frame b."""
        key = f"{cam_idx:02d}_{t:05d}"
        if key in self._mem_cache:
            return self._mem_cache[key]
        cache_path = Path(CACHE_DIR, self.subject, key + ".npz")
        if cache_path.exists():
            try:
                d = np.load(cache_path)
                flow, conf = torch.from_numpy(d["flow"]), torch.from_numpy(d["conf"])
                self._mem_cache[key] = (flow, conf)
                return flow, conf
            except Exception:
                # another process was still writing this file (parallel runs on the same subject
                # share the cache dir) -- fall through and just recompute instead of crashing.
                pass

        self._ensure_raft()
        img_a, img_b = self._load_img(img_path_a), self._load_img(img_path_b)
        h, w = img_a.shape[1], img_a.shape[2]
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        pad = lambda im: F.pad(im[None], (0, pad_w, 0, pad_h), mode="replicate")[0]
        ta, tb = self._transforms(pad(img_a)[None], pad(img_b)[None])
        ta, tb = ta.to(self.device), tb.to(self.device)
        with torch.no_grad():
            flow_fwd = self._raft(ta, tb)[-1][0].cpu()[:, :h, :w]
            flow_bwd = self._raft(tb, ta)[-1][0].cpu()[:, :h, :w]

        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        sample_x = (xx + flow_fwd[0]).clamp(0, w - 1).long()
        sample_y = (yy + flow_fwd[1]).clamp(0, h - 1).long()
        fb_err = ((flow_fwd[0] + flow_bwd[0][sample_y, sample_x]) ** 2 +
                  (flow_fwd[1] + flow_bwd[1][sample_y, sample_x]) ** 2).sqrt()
        conf = 1.0 / (1.0 + fb_err)  # in (0, 1], higher = more trustworthy

        # atomic write: parallel runs on the same subject share this cache dir, and np.savez isn't
        # atomic on its own -- write to a per-process temp file, then os.replace() (atomic on POSIX)
        # so a concurrent reader never sees a partially-written file.
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp_path, flow=flow_fwd.numpy(), conf=conf.numpy())
        os.replace(tmp_path, cache_path)
        self._mem_cache[key] = (flow_fwd, conf)
        return flow_fwd, conf


def sample_flow_at(flow, pixels):
    """flow: (2, H, W) tensor. pixels: (N, 2) float pixel coords (x, y). Returns (N, 2) bilinearly
    sampled flow vectors, via grid_sample (needs pixels normalized to [-1, 1])."""
    _, H, W = flow.shape
    gx = (pixels[:, 0] / (W - 1)) * 2 - 1
    gy = (pixels[:, 1] / (H - 1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)[None, :, None, :]  # (1, N, 1, 2)
    sampled = F.grid_sample(flow[None], grid, align_corners=True, mode="bilinear")  # (1, 2, N, 1)
    return sampled[0, :, :, 0].transpose(0, 1)  # (N, 2)


def hair_xyz_offset_only(gaussians, hair_mask):
    """Same math as GaussianModel.get_xyz, but with the rigid/pose-dependent factors (face_orien_mat,
    face_scaling, face_center, and the per-Gaussian raw _xyz) detached, keeping gradient live only
    through strand_delta_cumsum (the actual thing this loss is meant to supervise). Using the plain
    get_xyz here would let the flow loss's gradient also push on FLAME pose (rotation/neck_pose/
    jaw_pose/translation are all trainable) for both sampled timesteps -- a noisy, low-frequency
    auxiliary signal destabilizing a fit that's normally driven by the much more reliable per-frame
    photometric loss. Restricting the gradient path to the strand offset matches the original design
    (project Delta_j(t+1) - Delta_j(t), not the full position)."""
    binding = gaussians.binding[hair_mask]
    orien = gaussians.face_orien_mat[binding].detach()
    scaling = gaussians.face_scaling[binding].detach()
    center = gaussians.face_center[binding].detach()
    local_xyz = gaussians._xyz[hair_mask].detach()

    rigid = torch.bmm(orien, local_xyz[..., None]).squeeze(-1) * scaling + center

    strand_id = gaussians.face_strand_id[binding]
    chain_pos = gaussians.face_chain_pos[binding]
    local_delta = gaussians.strand_delta_cumsum[strand_id.clamp(min=0), chain_pos]  # (M, 3), gradient live
    global_delta = torch.bmm(orien, local_delta[..., None]).squeeze(-1) * scaling
    return rigid + global_delta


def compute_flow_supervision_loss(gaussians, cam_a, cam_b, flow_cache, min_confidence=0.3):
    """cam_a, cam_b: Camera objects for the same camera at adjacent timesteps t, t+1. Returns a
    scalar loss (or None if no valid hair Gaussians / image pair available)."""
    hair_mask = gaussians.face_strand_id[gaussians.binding] >= 0
    if not hair_mask.any():
        return None

    gaussians.select_mesh_by_timestep(cam_a.timestep)
    xyz_a = hair_xyz_offset_only(gaussians, hair_mask)
    gaussians.select_mesh_by_timestep(cam_b.timestep)
    xyz_b = hair_xyz_offset_only(gaussians, hair_mask)

    px_a = project_points_to_pixels(xyz_a, cam_a)  # (M, 2), differentiable w.r.t. strand params
    px_b = project_points_to_pixels(xyz_b, cam_b)
    predicted_flow = px_b - px_a  # (M, 2)

    flow_gt, conf = flow_cache.get(cam_a.image_path, cam_b.image_path,
                                    int(cam_a.image_name.split("_")[1]), cam_a.timestep)
    flow_gt, conf = flow_gt.to(px_a.device), conf.to(px_a.device)
    target_flow = sample_flow_at(flow_gt, px_a.detach())  # target itself doesn't need gradient
    weight = sample_flow_at(conf[None].expand(2, -1, -1), px_a.detach())[:, 0]  # reuse sampler, take one channel

    valid = weight > min_confidence
    if valid.sum() == 0:
        return None
    err = (predicted_flow - target_flow)[valid].norm(dim=-1)
    return (err * weight[valid]).sum() / weight[valid].sum()
