"""
A-0: optical-flow quality check on the hair region, before committing to full A-2 integration.

For each subject, finds same-camera adjacent-timestep GT frame pairs (already rendered under
output/free60k_baseline_{subj}/test/ours_60000/gt/), runs torchvision's pretrained RAFT-Large,
and reports:
  1. mean flow magnitude inside the hair-region crop vs. the rest of the frame
  2. forward-backward consistency error inside the hair crop vs. the rest of the frame
     (large gap here => RAFT is unreliable specifically where this project needs it)
Also saves one flow-visualization PNG per subject for a qualitative look.

Reuses the hair bbox-from-FLAME-landmark-projection logic already in precise_hair_crop_eval.py.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from flame_model.flame import FlameHead

PROJECT = "/home/hee/hee_data/GaussianAvatars"
SUBJECTS = ["264", "304", "218"]  # highest peakiness/best result, 2nd-highest/worst (anomaly), low/2nd-best
N_PAIRS_PER_SUBJECT = 6
DEVICE = "cuda"

flame = FlameHead(shape_params=300, expr_params=100).cuda()
hair_vid = flame.mask.v.hair.cpu().numpy()

weights = Raft_Large_Weights.DEFAULT
raft = raft_large(weights=weights).to(DEVICE).eval()
raft_transforms = weights.transforms()


def hair_bbox_for_frame(frame, data_dir, w, h, vert_cache, pad_frac=0.15):
    ts = frame["timestep_index"]
    flame_path = str(Path(data_dir) / frame["flame_param_path"])
    if flame_path not in vert_cache:
        # per-frame flame_param npz: shape is (300,) unbatched, everything else already has a
        # leading batch dim of 1 (translation/rotation/neck_pose/jaw_pose/eyes_pose/expr/static_offset)
        d = dict(np.load(flame_path, allow_pickle=True))
        static_offset = torch.from_numpy(d["static_offset"]).float().cuda()
        dynamic_offset = (
            torch.from_numpy(d["dynamic_offset"]).float().cuda()
            if "dynamic_offset" in d else torch.zeros_like(static_offset)
        )
        verts, _ = flame(
            torch.from_numpy(d["shape"]).float()[None].cuda(),
            torch.from_numpy(d["expr"]).float().cuda(),
            torch.from_numpy(d["rotation"]).float().cuda(),
            torch.from_numpy(d["neck_pose"]).float().cuda(),
            torch.from_numpy(d["jaw_pose"]).float().cuda(),
            torch.from_numpy(d["eyes_pose"]).float().cuda(),
            torch.from_numpy(d["translation"]).float().cuda(),
            zero_centered_at_root_node=False, return_landmarks=False, return_verts_cano=True,
            static_offset=static_offset,
            dynamic_offset=dynamic_offset,
        )
        vert_cache[flame_path] = verts[0].detach().cpu().numpy()
    verts_world = vert_cache[flame_path]
    hair_pts = verts_world[hair_vid]  # (357, 3)

    Rt = np.array(frame["transform_matrix"])
    fl_x, fl_y, cx, cy = frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]
    cam2world = Rt
    world2cam = np.linalg.inv(cam2world)
    pts_h = np.concatenate([hair_pts, np.ones((hair_pts.shape[0], 1))], axis=1)
    pts_cam = (world2cam @ pts_h.T).T[:, :3]
    valid = pts_cam[:, 2] < 0  # camera looks down -z in this convention (matches precise_hair_crop_eval.py)
    u = fl_x * (pts_cam[:, 0] / -pts_cam[:, 2]) + cx
    v = fl_y * (pts_cam[:, 1] / -pts_cam[:, 2]) + cy
    if valid.sum() < 5:
        return 0, 0, w, int(h * 0.35)
    u, v = u[valid], v[valid]
    x0, x1 = u.min(), u.max()
    y0, y1 = v.min(), v.max()
    pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    x0, x1 = max(0, x0 - pw), min(w, x1 + pw)
    y0, y1 = max(0, y0 - ph), min(h, y1 + ph)
    return int(x0), int(y0), int(x1), int(y1)


def load_img(path):
    # keep as uint8 -- torchvision's RAFT transform (convert_image_dtype) only rescales 0-255 -> 0-1
    # when the input dtype says so; a float tensor already in [0,255] gets treated as already-[0,1]
    # and produces garbage flow.
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).contiguous()


def flow_to_color(flow):
    # standard HSV-based flow visualization
    fx, fy = flow[0], flow[1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)
    h = (ang + np.pi) / (2 * np.pi)
    s = np.ones_like(h)
    v = np.clip(mag / (mag.max() + 1e-6), 0, 1)
    import colorsys
    hsv = np.stack([h, s, v], axis=-1)
    rgb = np.zeros_like(hsv)
    it = np.nditer(h, flags=['multi_index'])
    for _ in it:
        i, j = it.multi_index
        rgb[i, j] = colorsys.hsv_to_rgb(hsv[i, j, 0], hsv[i, j, 1], hsv[i, j, 2])
    return (rgb * 255).astype(np.uint8)


results = {}
for subj in SUBJECTS:
    data_dir = f"{PROJECT}/data/{subj}_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
    gt_dir = f"{PROJECT}/output/free60k_baseline_{subj}/test/ours_60000/gt"
    with open(f"{data_dir}/transforms_test.json") as f:
        meta = json.load(f)
    frames = meta["frames"]

    from collections import defaultdict
    by_cam = defaultdict(list)
    for i, fr in enumerate(frames):
        by_cam[fr["camera_index"]].append((fr["timestep_index"], i))
    cam0 = 0
    by_cam[cam0].sort()

    vert_cache = {}
    hair_mags, bg_mags = [], []
    hair_fb_err, bg_fb_err = [], []
    n_done = 0
    for k in range(len(by_cam[cam0]) - 1):
        ts_a, i_a = by_cam[cam0][k]
        ts_b, i_b = by_cam[cam0][k + 1]
        if ts_b - ts_a != 1:
            continue
        fa, fb = frames[i_a], frames[i_b]
        pa, pb = Path(gt_dir) / f"{i_a:05d}.png", Path(gt_dir) / f"{i_b:05d}.png"
        if not (pa.exists() and pb.exists()):
            continue
        img_a, img_b = load_img(pa), load_img(pb)
        h, w = img_a.shape[1], img_a.shape[2]
        x0, y0, x1, y1 = hair_bbox_for_frame(fa, data_dir, w, h, vert_cache)
        if x1 <= x0 or y1 <= y0:
            continue

        # RAFT requires H, W divisible by 8 -- pad up, run, then crop the flow back down
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        pad_img = lambda im: F.pad(im[None], (0, pad_w, 0, pad_h), mode="replicate")[0]

        ta, tb = raft_transforms(pad_img(img_a)[None], pad_img(img_b)[None])
        ta, tb = ta.to(DEVICE), tb.to(DEVICE)
        with torch.no_grad():
            flow_fwd = raft(ta, tb)[-1][0].cpu().numpy()  # (2, H', W') at model's internal resolution
            flow_bwd = raft(tb, ta)[-1][0].cpu().numpy()
        flow_fwd = flow_fwd[:, :h, :w]
        flow_bwd = flow_bwd[:, :h, :w]
        # resize flow back to original image resolution if RAFT changed it (torchvision keeps it in general)
        if flow_fwd.shape[1:] != (h, w):
            flow_fwd_t = F.interpolate(torch.from_numpy(flow_fwd)[None], size=(h, w), mode="bilinear")[0].numpy()
            flow_bwd_t = F.interpolate(torch.from_numpy(flow_bwd)[None], size=(h, w), mode="bilinear")[0].numpy()
        else:
            flow_fwd_t, flow_bwd_t = flow_fwd, flow_bwd

        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True

        mag = np.sqrt(flow_fwd_t[0] ** 2 + flow_fwd_t[1] ** 2)
        hair_mags.append(mag[mask].mean())
        bg_mags.append(mag[~mask].mean())

        # forward-backward consistency: warp flow_bwd by flow_fwd and add, should be ~0 if consistent
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        sample_x = np.clip(xx + flow_fwd_t[0], 0, w - 1).astype(np.int32)
        sample_y = np.clip(yy + flow_fwd_t[1], 0, h - 1).astype(np.int32)
        fb_err = np.sqrt(
            (flow_fwd_t[0] + flow_bwd_t[0][sample_y, sample_x]) ** 2 +
            (flow_fwd_t[1] + flow_bwd_t[1][sample_y, sample_x]) ** 2
        )
        hair_fb_err.append(fb_err[mask].mean())
        bg_fb_err.append(fb_err[~mask].mean())

        if n_done == 0:
            vis = flow_to_color(flow_fwd_t)
            Image.fromarray(vis).save(f"{PROJECT}/dev/hair_avatars/flow_vis_{subj}_t{ts_a}.png")
            crop_a = img_a.permute(1, 2, 0).numpy()[y0:y1, x0:x1].astype(np.uint8)
            Image.fromarray(crop_a).save(f"{PROJECT}/dev/hair_avatars/flow_vis_{subj}_t{ts_a}_haircrop_gt.png")
            crop_flow = vis[y0:y1, x0:x1]
            Image.fromarray(crop_flow).save(f"{PROJECT}/dev/hair_avatars/flow_vis_{subj}_t{ts_a}_haircrop_flow.png")

        n_done += 1
        if n_done >= N_PAIRS_PER_SUBJECT:
            break

    results[subj] = dict(
        n_pairs=n_done,
        hair_mag_mean=float(np.mean(hair_mags)) if hair_mags else None,
        bg_mag_mean=float(np.mean(bg_mags)) if bg_mags else None,
        hair_fb_err_mean=float(np.mean(hair_fb_err)) if hair_fb_err else None,
        bg_fb_err_mean=float(np.mean(bg_fb_err)) if bg_fb_err else None,
    )
    print(f"subject {subj}: n_pairs={n_done}")
    print(f"  flow magnitude   hair={results[subj]['hair_mag_mean']:.3f}px  bg={results[subj]['bg_mag_mean']:.3f}px")
    print(f"  fwd-bwd err      hair={results[subj]['hair_fb_err_mean']:.3f}px  bg={results[subj]['bg_fb_err_mean']:.3f}px"
          f"  (ratio hair/bg = {results[subj]['hair_fb_err_mean']/max(results[subj]['bg_fb_err_mean'],1e-6):.2f})")

with open(f"{PROJECT}/dev/hair_avatars/optical_flow_check_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved summary to dev/hair_avatars/optical_flow_check_summary.json")
