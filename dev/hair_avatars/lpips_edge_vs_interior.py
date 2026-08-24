"""
Test hypothesis: static-only's LPIPS regression (despite better L1/PSNR everywhere) comes
from perceptual sensitivity to small geometric shifts concentrated at hair-strand edges /
silhouette boundaries, which L1 (a flat average) barely registers but LPIPS (CNN-feature-based,
receptive-field-local) penalizes heavily.

Method: for each test frame, build an edge mask (Canny on GT, dilated) covering hair-curl
boundaries + silhouette. Sample patches centered on edge-mask pixels ("edge" set) and centered
on non-edge pixels ("interior" set). Compute LPIPS(patch_vs_gt) for baseline and static-only
separately in each set, then compare the mean shift.
"""
import os, glob, random
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from lpipsPyTorch import lpips

PROJECT = "/hdd2/hee_data/GaussianAvatars"
BASE_DIR = f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000"
STATIC_DIR = f"{PROJECT}/output/staticonly60k_306/test/ours_60000"

PATCH = 64
N_PATCHES_PER_FRAME = 24
N_FRAMES = 12
random.seed(0)


def load(d, f):
    return np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), dtype=np.uint8)


def to_tensor(patch):
    t = torch.from_numpy(patch).float().permute(2, 0, 1) / 255.0
    return (t * 2 - 1).unsqueeze(0).cuda()


def edge_mask(gt):
    gray = (gt.astype(np.float32) @ np.array([0.299, 0.587, 0.114])) / 255.0
    gray = ndimage.gaussian_filter(gray, sigma=1.0)
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    thresh = np.percentile(mag, 92)  # top ~8% gradient magnitude pixels = strand/silhouette edges
    edges = mag > thresh
    edges = ndimage.binary_dilation(edges, structure=np.ones((7, 7)))
    return edges


def sample_centers(mask, n, h, w, margin):
    ys, xs = np.where(mask)
    valid = (ys > margin) & (ys < h - margin) & (xs > margin) & (xs < w - margin)
    ys, xs = ys[valid], xs[valid]
    if len(ys) == 0:
        return []
    idx = np.random.choice(len(ys), size=min(n, len(ys)), replace=False)
    return list(zip(ys[idx], xs[idx]))


gt_files = sorted(os.listdir(f"{BASE_DIR}/gt"))
frames = random.sample(gt_files, min(N_FRAMES, len(gt_files)))

edge_base, edge_static = [], []
interior_base, interior_static = [], []

margin = PATCH // 2 + 2
for fname in frames:
    gt = load(f"{BASE_DIR}/gt", fname)
    base = load(f"{BASE_DIR}/renders", fname)
    static = load(f"{STATIC_DIR}/renders", fname)
    h, w = gt.shape[:2]

    em = edge_mask(gt)
    im = ~ndimage.binary_dilation(em, structure=np.ones((15, 15)))

    edge_centers = sample_centers(em, N_PATCHES_PER_FRAME, h, w, margin)
    interior_centers = sample_centers(im, N_PATCHES_PER_FRAME, h, w, margin)

    def crop(img, cy, cx):
        return img[cy - PATCH // 2:cy + PATCH // 2, cx - PATCH // 2:cx + PATCH // 2]

    with torch.no_grad():
        for cy, cx in edge_centers:
            gt_p, base_p, static_p = crop(gt, cy, cx), crop(base, cy, cx), crop(static, cy, cx)
            edge_base.append(lpips(to_tensor(base_p), to_tensor(gt_p), net_type='vgg').item())
            edge_static.append(lpips(to_tensor(static_p), to_tensor(gt_p), net_type='vgg').item())
        for cy, cx in interior_centers:
            gt_p, base_p, static_p = crop(gt, cy, cx), crop(base, cy, cx), crop(static, cy, cx)
            interior_base.append(lpips(to_tensor(base_p), to_tensor(gt_p), net_type='vgg').item())
            interior_static.append(lpips(to_tensor(static_p), to_tensor(gt_p), net_type='vgg').item())

eb, es = np.array(edge_base), np.array(edge_static)
ib, is_ = np.array(interior_base), np.array(interior_static)

print(f"frames used: {len(frames)}")
print(f"\n[EDGE region]     n={len(eb)}   baseline={eb.mean():.5f}  static={es.mean():.5f}  delta={es.mean()-eb.mean():+.5f}  (static {'WORSE' if es.mean()>eb.mean() else 'better'})")
print(f"[INTERIOR region]  n={len(ib)}   baseline={ib.mean():.5f}  static={is_.mean():.5f}  delta={is_.mean()-ib.mean():+.5f}  (static {'WORSE' if is_.mean()>ib.mean() else 'better'})")
