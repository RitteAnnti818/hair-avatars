"""
Precise hair-region evaluation: project the actual FLAME hair vertices (357 verts) into each
test camera's image plane to get a real hair bounding box, instead of the crude "top 35%" proxy.
Compares baseline vs hair-strand renders (both at 60k iterations, subject 306).
"""
import json, os
import numpy as np
import torch
from PIL import Image
from flame_model.flame import FlameHead

PROJECT = "/hdd2/hee_data/GaussianAvatars"
DATA_DIR = f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
BASE_DIR = f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000"
HAIR_DIR = f"{PROJECT}/output/fair60k_hairstrand_306/test/ours_60000"

flame = FlameHead(shape_params=300, expr_params=100).cuda()
hair_vid = flame.mask.v.hair.cpu().numpy()  # (357,)

with open(f"{DATA_DIR}/transforms_test.json") as f:
    meta = json.load(f)
frames = meta["frames"]

# cache per-timestep flame_param -> vertices (only for timesteps actually used in test)
_vert_cache = {}
def get_hair_verts_2d(frame):
    ts = frame["timestep_index"]
    flame_path = os.path.join(DATA_DIR, frame["flame_param_path"])
    if flame_path not in _vert_cache:
        d = dict(np.load(flame_path, allow_pickle=True))
        with torch.no_grad():
            out = flame(
                torch.from_numpy(d["shape"]).float().cuda()[None, ...] if d["shape"].ndim == 1 else torch.from_numpy(d["shape"]).float().cuda(),
                torch.from_numpy(d["expr"]).float().cuda(),
                torch.from_numpy(d["rotation"]).float().cuda(),
                torch.from_numpy(d["neck_pose"]).float().cuda(),
                torch.from_numpy(d["jaw_pose"]).float().cuda(),
                torch.from_numpy(d["eyes_pose"]).float().cuda(),
                torch.from_numpy(d["translation"]).float().cuda(),
                zero_centered_at_root_node=False, return_landmarks=False, return_verts_cano=False,
                static_offset=torch.from_numpy(d["static_offset"]).float().cuda(),
            )
        verts = out[0] if isinstance(out, (list, tuple)) else out
        _vert_cache[flame_path] = verts[0].detach().cpu().numpy()  # (V,3)
    verts = _vert_cache[flame_path]
    hair_pts_world = verts[hair_vid]  # (357, 3)

    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([hair_pts_world, np.ones((hair_pts_world.shape[0], 1))], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]

    fx, fy, cx, cy = frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]
    z = np.clip(pts_cam[:, 2], 1e-4, None)
    u = fx * pts_cam[:, 0] / z + cx
    v = fy * pts_cam[:, 1] / z + cy
    return u, v


def bbox_for_frame(frame, w, h, pad_frac=0.15):
    u, v = get_hair_verts_2d(frame)
    valid = (u > -w) & (u < 2 * w) & (v > -h) & (v < 2 * h)
    if valid.sum() < 5:
        return 0, 0, w, int(h * 0.35)  # fallback to old proxy
    u, v = u[valid], v[valid]
    x0, x1 = u.min(), u.max()
    y0, y1 = v.min(), v.max()
    pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0, x1 = max(0, x0 - pw), min(w, x1 + pw)
    y0, y1 = max(0, y0 - ph), min(h, y1 + ph)
    return int(x0), int(y0), int(x1), int(y1)


def load(d, f):
    return np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), dtype=np.float32) / 255.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else -10 * np.log10(mse)


records = []
for i, frame in enumerate(frames):
    fname = f"{i:05d}.png"
    gt = load(f"{BASE_DIR}/gt", fname)
    base = load(f"{BASE_DIR}/renders", fname)
    hair = load(f"{HAIR_DIR}/renders", fname)
    h, w = gt.shape[:2]
    x0, y0, x1, y1 = bbox_for_frame(frame, w, h)
    if x1 <= x0 or y1 <= y0:
        continue
    gt_c, base_c, hair_c = gt[y0:y1, x0:x1], base[y0:y1, x0:x1], hair[y0:y1, x0:x1]

    l1_base = np.abs(gt_c - base_c).mean()
    l1_hair = np.abs(gt_c - hair_c).mean()
    psnr_base = psnr(gt_c, base_c)
    psnr_hair = psnr(gt_c, hair_c)
    records.append({
        "idx": i, "fname": fname, "timestep": frame["timestep_index"], "camera": frame["camera_index"],
        "bbox": (x0, y0, x1, y1),
        "l1_base": float(l1_base), "l1_hair": float(l1_hair),
        "psnr_base": float(psnr_base), "psnr_hair": float(psnr_hair),
    })
    if i % 300 == 0:
        print(f"  {i}/{len(frames)}")

l1_base_all = np.array([r["l1_base"] for r in records])
l1_hair_all = np.array([r["l1_hair"] for r in records])
psnr_base_all = np.array([r["psnr_base"] for r in records])
psnr_hair_all = np.array([r["psnr_hair"] for r in records])

print(f"\n=== Precise FLAME-hair-mask crop, N={len(records)} frames ===")
print(f"L1:   base={l1_base_all.mean():.5f}  ours={l1_hair_all.mean():.5f}  ({(1 - l1_hair_all.mean()/l1_base_all.mean())*100:+.2f}%)")
print(f"PSNR: base={psnr_base_all.mean():.3f}  ours={psnr_hair_all.mean():.3f}  ({psnr_hair_all.mean()-psnr_base_all.mean():+.3f} dB)")

delta_psnr = psnr_hair_all - psnr_base_all  # positive = ours better
order = np.argsort(delta_psnr)
print(f"\nPer-frame PSNR delta (ours-base): mean={delta_psnr.mean():+.3f}  std={delta_psnr.std():.3f}")
print(f"Frames where ours is better: {(delta_psnr > 0).sum()}/{len(delta_psnr)} ({(delta_psnr>0).mean()*100:.1f}%)")

print("\n=== Top-5 BEST for ours (ours >> base) ===")
for i in order[-5:][::-1]:
    r = records[i]
    print(f"  {r['fname']} t={r['timestep']} cam={r['camera']}: base={r['psnr_base']:.2f} ours={r['psnr_hair']:.2f} (delta={delta_psnr[i]:+.2f})")

print("\n=== Top-5 WORST for ours (base >> ours) ===")
for i in order[:5]:
    r = records[i]
    print(f"  {r['fname']} t={r['timestep']} cam={r['camera']}: base={r['psnr_base']:.2f} ours={r['psnr_hair']:.2f} (delta={delta_psnr[i]:+.2f})")

with open(f"{PROJECT}/dev/hair_avatars/precise_hair_crop_records.json", "w") as f:
    json.dump(records, f, indent=1)
print(f"\nsaved {len(records)} records to dev/hair_avatars/precise_hair_crop_records.json")
