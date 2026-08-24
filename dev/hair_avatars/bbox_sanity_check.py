import json, os
import numpy as np
import torch
from PIL import Image, ImageDraw
from flame_model.flame import FlameHead

PROJECT = "/hdd2/hee_data/GaussianAvatars"
flame = FlameHead(shape_params=300, expr_params=100).cuda()
hair_vid = flame.mask.v.hair.cpu().numpy()

SUBJECTS = {
    "104": dict(data_dir=f"{PROJECT}/data/UNION10_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
                base_dir=f"{PROJECT}/output/UNION10EMOEXP_104_eval_600k/test/ours_60000"),
    "253": dict(data_dir=f"{PROJECT}/data/UNION10_253_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
                base_dir=f"{PROJECT}/output/fair60k_baseline_253/test/ours_60000"),
}

for sid, cfg in SUBJECTS.items():
    with open(f"{cfg['data_dir']}/transforms_test.json") as f:
        frames = json.load(f)["frames"]
    frame = frames[0]
    d = dict(np.load(os.path.join(cfg["data_dir"], frame["flame_param_path"]), allow_pickle=True))
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
    verts = (out[0] if isinstance(out, (list, tuple)) else out)[0].detach().cpu().numpy()
    hair_pts = verts[hair_vid]

    c2w = np.array(frame["transform_matrix"], dtype=np.float64)
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([hair_pts, np.ones((hair_pts.shape[0], 1))], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    fx, fy, cx, cy = frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]
    z = np.clip(pts_cam[:, 2], 1e-4, None)
    u = fx * pts_cam[:, 0] / z + cx
    v = fy * pts_cam[:, 1] / z + cy

    im = Image.open(f"{cfg['base_dir']}/gt/00000.png").convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)
    for uu, vv in zip(u, v):
        draw.ellipse([uu-2, vv-2, uu+2, vv+2], fill="lime")
    valid = (u > -w) & (u < 2*w) & (v > -h) & (v < 2*h)
    x0,x1 = u[valid].min(), u[valid].max()
    y0,y1 = v[valid].min(), v[valid].max()
    pad_w, pad_h = (x1-x0)*0.15, (y1-y0)*0.15
    draw.rectangle([x0-pad_w,y0-pad_h,x1+pad_w,y1+pad_h], outline="red", width=4)
    out_path = f"{PROJECT}/dev/hair_avatars/bboxcheck_{sid}.png"
    im.save(out_path)
    print(f"saved {out_path}")
