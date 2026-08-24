import argparse
import glob
import json
from pathlib import Path
from statistics import mean

SPLITS = [("test", "", "SR (self-reenactment)"), ("val", "_val", "NVS (novel-view)")]
NET_TYPES = [("vgg", ""), ("alex", "_alex")]


def load_metrics(scene_dir, split_suffix, net_suffix):
    fname = Path(scene_dir) / f"results{split_suffix}{net_suffix}.json"
    if not fname.exists():
        return None
    data = json.loads(fname.read_text())
    if not data:
        return None
    key = max(data.keys(), key=lambda k: int(k.rsplit("_", 1)[-1]))
    return data[key]


def main():
    parser = argparse.ArgumentParser(description="Aggregate results*.json across subjects into a summary table")
    parser.add_argument("--model_paths", "-m", nargs="+", default=None)
    parser.add_argument("--out", default="output/aggregate_summary.json")
    args = parser.parse_args()

    scene_dirs = args.model_paths or sorted(glob.glob("output/UNION10EMOEXP_*_eval_600k"))
    print(f"Aggregating over {len(scene_dirs)} scenes\n")

    summary = {}
    per_subject = {}
    for split, split_suffix, _ in SPLITS:
        summary[split] = {}
        per_subject[split] = {}
        for net_type, net_suffix in NET_TYPES:
            values = {"PSNR": [], "SSIM": [], "LPIPS": []}
            rows = {}
            for scene_dir in scene_dirs:
                subj = Path(scene_dir).name
                m = load_metrics(scene_dir, split_suffix, net_suffix)
                rows[subj] = m
                if m is not None:
                    for k in values:
                        values[k].append(m[k])
            summary[split][net_type] = {k: (mean(v) if v else None) for k, v in values.items()}
            per_subject[split][net_type] = rows

    print("=" * 74)
    print(f"{'Split':<25}{'PSNR':>9}{'SSIM':>9}{'LPIPS(vgg)':>15}{'LPIPS(alex)':>15}")
    print("-" * 74)
    for split, _, label in SPLITS:
        vgg, alex = summary[split]["vgg"], summary[split]["alex"]
        psnr = vgg["PSNR"] if vgg["PSNR"] is not None else float("nan")
        ssim = vgg["SSIM"] if vgg["SSIM"] is not None else float("nan")
        lp_vgg = vgg["LPIPS"] if vgg["LPIPS"] is not None else float("nan")
        lp_alex = alex["LPIPS"] if alex["LPIPS"] is not None else float("nan")
        print(f"{label:<25}{psnr:>9.2f}{ssim:>9.4f}{lp_vgg:>15.4f}{lp_alex:>15.4f}")
    print("=" * 74)

    for split, _, label in SPLITS:
        print(f"\n--- {label}: per-subject (PSNR / SSIM / LPIPS-vgg / LPIPS-alex) ---")
        rows_vgg = per_subject[split]["vgg"]
        rows_alex = per_subject[split]["alex"]
        for scene_dir in scene_dirs:
            subj = Path(scene_dir).name
            mv, ma = rows_vgg[subj], rows_alex[subj]
            if mv is None:
                print(f"  {subj:<35} MISSING")
                continue
            lp_alex = ma["LPIPS"] if ma is not None else float("nan")
            print(f"  {subj:<35} {mv['PSNR']:>7.2f}  {mv['SSIM']:>6.4f}  {mv['LPIPS']:>7.4f}  {lp_alex:>7.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_subject": per_subject}, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
