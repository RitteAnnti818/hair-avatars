"""
Strand Offset Transfer demo: swap subject 306's learned _strand_delta_static (hair shape
correction) onto subject 074's avatar identity, and vice versa. Both subjects share the same
K=32 strand topology (clustered on the FLAME canonical template, not per-subject), so the
tensors are directly swappable in shape/semantics. No training involved -- reuses existing
static-only checkpoints (output/cohthresh_02_306, output/staticonly60k_074).

Usage: PYTHONPATH=. python dev/hair_avatars/strand_transfer_demo.py
"""
import os
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser, Namespace

from scene import Scene
from scene.flame_gaussian_model import FlameGaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state

OUT_DIR = Path("dev/hair_avatars/strand_transfer_demo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = {
    "306": ("output/cohthresh_02_306",
            "data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"),
    "074": ("output/staticonly60k_074",
            "data/UNION10_074_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"),
}
ITERATION = 60000
FRAME_INDICES = [0, 300, 600, 900, 1200]  # spread across the test sequence for variety


def load_cfg_args(model_path):
    with open(os.path.join(model_path, "cfg_args")) as f:
        return eval(f.read())  # Namespace(...) literal, matches render.py's get_combined_args behavior


def build_gaussians_and_scene(model_path, source_path):
    cfg = load_cfg_args(model_path)
    # older cfg_args (pre-migration) recorded a stale absolute /hdd2/... source_path; override with
    # the current, correct local data directory instead of trusting the stored value.
    cfg.source_path = os.path.abspath(source_path)
    parser = ArgumentParser()
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    args = parser.parse_args([])
    for k, v in vars(cfg).items():
        setattr(args, k, v)
    args.model_path = model_path
    dataset = lp.extract(args)
    pipeline = pp.extract(args)

    gaussians = FlameGaussianModel(
        dataset.sh_degree,
        enable_hair_strands=getattr(dataset, "enable_hair_strands", False),
        strand_json_path=getattr(dataset, "strand_json_path", ""),
        disable_strand_dynamic=getattr(dataset, "disable_strand_dynamic", False),
    )
    scene = Scene(dataset, gaussians, load_iteration=ITERATION, shuffle=False)
    return dataset, pipeline, gaussians, scene


def render_frames(dataset, pipeline, gaussians, scene, tag):
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    test_cams = scene.getTestCameras()
    for fi in FRAME_INDICES:
        if fi >= len(test_cams):
            continue
        view = test_cams[fi]
        gaussians.select_mesh_by_timestep(view.timestep)
        with torch.no_grad():
            image = render(view, gaussians, pipeline, background)["render"]
        arr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        from PIL import Image as PILImage
        PILImage.fromarray(arr).save(OUT_DIR / f"{tag}_frame{fi:05d}.png")
    print(f"[{tag}] saved {len(FRAME_INDICES)} frames to {OUT_DIR}")


if __name__ == "__main__":
    safe_state(True)

    print("=== loading subject 306 (own hair) ===")
    dataset_306, pipeline_306, gaussians_306, scene_306 = build_gaussians_and_scene(*SUBJECTS["306"])
    static_306 = gaussians_306._strand_delta_static.clone()
    render_frames(dataset_306, pipeline_306, gaussians_306, scene_306, "306_own")

    print("=== loading subject 074 (own hair) ===")
    dataset_074, pipeline_074, gaussians_074, scene_074 = build_gaussians_and_scene(*SUBJECTS["074"])
    static_074 = gaussians_074._strand_delta_static.clone()
    render_frames(dataset_074, pipeline_074, gaussians_074, scene_074, "074_own")

    assert static_306.shape == static_074.shape, f"shape mismatch: {static_306.shape} vs {static_074.shape}"
    print(f"strand_delta_static shape (shared K=32 topology): {static_306.shape}")

    print("=== 306 identity + 074's hair-shape correction ===")
    gaussians_306._strand_delta_static = static_074.clone()
    render_frames(dataset_306, pipeline_306, gaussians_306, scene_306, "306_wearing_074hair")

    print("=== 074 identity + 306's hair-shape correction ===")
    gaussians_074._strand_delta_static = static_306.clone()
    render_frames(dataset_074, pipeline_074, gaussians_074, scene_074, "074_wearing_306hair")

    print("done. Compare *_own vs *_wearing_* pairs visually in", OUT_DIR)
