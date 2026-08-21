"""
Investigate why our 600k reproduction's full-image LPIPS (0.1298 avg over 8 subjects) is much
higher than the paper's reported self-reenactment average (0.076), despite PSNR/SSIM matching the
paper almost exactly. Hypothesis: LPIPS is disproportionately driven by the person/background
silhouette boundary (consistent with what we found repeatedly in the hair-strand LPIPS investigations),
so a small boundary/antialiasing mismatch could inflate LPIPS specifically without hurting PSNR/SSIM.

Method: for a sample of test frames, build an edge mask via gradient magnitude on GT, and compare
patch-LPIPS(render, GT) sampled at edge-mask locations vs interior locations.
"""
import os, random
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from lpipsPyTorch import lpips

PROJECT = "/hdd2/hee_data/GaussianAvatars"
SUBJECT_DIR = f"{PROJECT}/output/UNION10EMOEXP_306_eval_600k/test/ours_600000"

PATCH = 64
N_PATCHES_PER_FRAME = 30
N_FRAMES = 15
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
    thresh = np.percentile(mag, 92)
    edges = mag > thresh
    edges = ndimage.binary_dilation(edges, structure=np.ones((7, 7)))
    return edges


def silhouette_boundary_mask(gt):
    """Specifically: the person-vs-white-background boundary (near-white pixels adjacent to non-white)."""
    is_bg = (gt.astype(np.float32).mean(axis=-1) > 250)
    boundary = ndimage.binary_dilation(is_bg, structure=np.ones((5, 5))) & ~is_bg
    boundary = ndimage.binary_dilation(boundary, structure=np.ones((9, 9)))
    return boundary, is_bg


def sample_centers(mask, n, h, w, margin):
    ys, xs = np.where(mask)
    valid = (ys > margin) & (ys < h - margin) & (xs > margin) & (xs < w - margin)
    ys, xs = ys[valid], xs[valid]
    if len(ys) == 0:
        return []
    idx = np.random.choice(len(ys), size=min(n, len(ys)), replace=False)
    return list(zip(ys[idx], xs[idx]))


gt_files = sorted(os.listdir(f"{SUBJECT_DIR}/gt"))
frames = random.sample(gt_files, min(N_FRAMES, len(gt_files)))

edge_scores, interior_scores = [], []
silbound_scores, sil_interior_scores = [], []
margin = PATCH // 2 + 2

with torch.no_grad():
    for fname in frames:
        gt = load(f"{SUBJECT_DIR}/gt", fname)
        render = load(f"{SUBJECT_DIR}/renders", fname)
        h, w = gt.shape[:2]

        em = edge_mask(gt)
        im = ~ndimage.binary_dilation(em, structure=np.ones((15, 15)))
        silb, is_bg = silhouette_boundary_mask(gt)
        sil_int = ~ndimage.binary_dilation(silb, structure=np.ones((15, 15))) & ~is_bg

        def crop(img, cy, cx):
            return img[cy - PATCH // 2:cy + PATCH // 2, cx - PATCH // 2:cx + PATCH // 2]

        for cy, cx in sample_centers(em, N_PATCHES_PER_FRAME, h, w, margin):
            gt_p, r_p = crop(gt, cy, cx), crop(render, cy, cx)
            edge_scores.append(lpips(to_tensor(r_p), to_tensor(gt_p), net_type='vgg').item())
        for cy, cx in sample_centers(im, N_PATCHES_PER_FRAME, h, w, margin):
            gt_p, r_p = crop(gt, cy, cx), crop(render, cy, cx)
            interior_scores.append(lpips(to_tensor(r_p), to_tensor(gt_p), net_type='vgg').item())
        for cy, cx in sample_centers(silb, N_PATCHES_PER_FRAME, h, w, margin):
            gt_p, r_p = crop(gt, cy, cx), crop(render, cy, cx)
            silbound_scores.append(lpips(to_tensor(r_p), to_tensor(gt_p), net_type='vgg').item())
        for cy, cx in sample_centers(sil_int, N_PATCHES_PER_FRAME, h, w, margin):
            gt_p, r_p = crop(gt, cy, cx), crop(render, cy, cx)
            sil_interior_scores.append(lpips(to_tensor(r_p), to_tensor(gt_p), net_type='vgg').item())

edge_scores, interior_scores = np.array(edge_scores), np.array(interior_scores)
silbound_scores, sil_interior_scores = np.array(silbound_scores), np.array(sil_interior_scores)

print(f"frames used: {len(frames)}")
print(f"\n[general high-gradient edges]  n_edge={len(edge_scores)} mean={edge_scores.mean():.5f}   "
      f"n_interior={len(interior_scores)} mean={interior_scores.mean():.5f}   "
      f"ratio(edge/interior)={edge_scores.mean()/interior_scores.mean():.2f}x")
print(f"[person/bg silhouette boundary] n_bound={len(silbound_scores)} mean={silbound_scores.mean():.5f}   "
      f"n_interior(fg only)={len(sil_interior_scores)} mean={sil_interior_scores.mean():.5f}   "
      f"ratio(boundary/interior)={silbound_scores.mean()/sil_interior_scores.mean():.2f}x")

# whole-image LPIPS for reference
whole_scores = []
with torch.no_grad():
    for fname in frames:
        gt = load(f"{SUBJECT_DIR}/gt", fname)
        render = load(f"{SUBJECT_DIR}/renders", fname)
        gt_t = torch.from_numpy(gt).float().permute(2,0,1)/255.0
        r_t = torch.from_numpy(render).float().permute(2,0,1)/255.0
        gt_t = (gt_t*2-1).unsqueeze(0).cuda()
        r_t = (r_t*2-1).unsqueeze(0).cuda()
        whole_scores.append(lpips(r_t, gt_t, net_type='vgg').item())
whole_scores = np.array(whole_scores)
print(f"\n[whole-image LPIPS, same {len(frames)} frames] mean={whole_scores.mean():.5f}")
