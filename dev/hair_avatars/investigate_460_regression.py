"""
460 is the only subject where static-only (coherence-on, threshold=0.1) regressed vs baseline
in the precise hair-crop metric (-0.053dB, only 34.3% of frames better). Investigate whether
this is a few outlier frames or a systematic effect, and save a visual comparison of the worst case.
"""
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = "/hdd2/hee_data/GaussianAvatars"
BASE_DIR = f"{PROJECT}/output/fair60k_baseline_460/test/ours_60000"
STATIC_DIR = f"{PROJECT}/output/staticonly60k_460/test/ours_60000"


def load(d, f):
    return np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), dtype=np.float32) / 255.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else -10 * np.log10(mse)


fnames = sorted(os.listdir(f"{BASE_DIR}/gt"))
records = []
for fname in fnames:
    gt = load(f"{BASE_DIR}/gt", fname)
    base = load(f"{BASE_DIR}/renders", fname)
    static = load(f"{STATIC_DIR}/renders", fname)
    records.append({
        "fname": fname,
        "psnr_base": psnr(gt, base),
        "psnr_static": psnr(gt, static),
    })

deltas = np.array([r["psnr_static"] - r["psnr_base"] for r in records])
print(f"N={len(records)}  mean_delta={deltas.mean():+.4f}  std={deltas.std():.4f}  "
      f"pct_better={(deltas>0).mean()*100:.1f}%")
print(f"delta range: min={deltas.min():.3f}  max={deltas.max():.3f}")
print(f"percentiles: p10={np.percentile(deltas,10):.3f}  p50={np.percentile(deltas,50):.3f}  p90={np.percentile(deltas,90):.3f}")

order = np.argsort(deltas)
worst = [records[i] for i in order[:5]]
best = [records[i] for i in order[-5:]]
print("\nworst 5 frames (static much worse than baseline):")
for r in worst:
    print(f"  {r['fname']}: base={r['psnr_base']:.3f} static={r['psnr_static']:.3f} delta={r['psnr_static']-r['psnr_base']:+.3f}")
print("\nbest 5 frames (static much better):")
for r in best:
    print(f"  {r['fname']}: base={r['psnr_base']:.3f} static={r['psnr_static']:.3f} delta={r['psnr_static']-r['psnr_base']:+.3f}")

# visualize the single worst frame
worst_fname = records[order[0]]["fname"]
gt = load(f"{BASE_DIR}/gt", worst_fname)
base = load(f"{BASE_DIR}/renders", worst_fname)
static = load(f"{STATIC_DIR}/renders", worst_fname)
diff = np.abs(base - static).mean(axis=-1)

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
for ax, img, title in zip(axes[:3], [gt, base, static], ["GT", "baseline", f"static-only (worst: {worst_fname})"]):
    ax.imshow(img); ax.set_title(title); ax.axis("off")
im = axes[3].imshow(diff, cmap="inferno"); axes[3].set_title("|baseline - static|"); axes[3].axis("off")
plt.colorbar(im, ax=axes[3], fraction=0.046)
plt.tight_layout()
plt.savefig(f"{PROJECT}/dev/hair_avatars/460_worst_frame_check.png", dpi=100)
print(f"\nsaved visualization to dev/hair_avatars/460_worst_frame_check.png (worst frame: {worst_fname})")
