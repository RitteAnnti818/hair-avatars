#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :])
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :])
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, split="test", batch_size=1, net_type="vgg"):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / split

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for start in tqdm(range(0, len(renders), batch_size), desc="Metric evaluation progress"):
                    chunk_r = renders[start:start + batch_size]
                    chunk_g = gts[start:start + batch_size]
                    try:
                        render = torch.cat(chunk_r, 0).cuda()
                        gt = torch.cat(chunk_g, 0).cuda()
                    except RuntimeError:
                        # mismatched resolutions within the batch -- fall back to one-by-one for this chunk
                        for r, g in zip(chunk_r, chunk_g):
                            r, g = r.cuda(), g.cuda()
                            ssims.append(ssim(r, g).item())
                            psnrs.append(psnr(r, g).item())
                            lpipss.append(lpips(r, g, net_type=net_type).item())
                        continue

                    ssims.extend(ssim(render, gt, size_average=False).tolist())
                    psnrs.extend(psnr(render, gt).flatten().tolist())
                    lpipss.extend(lpips(render, gt, net_type=net_type).flatten().tolist())
                    del render, gt
                    torch.cuda.empty_cache()

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            suffix = "" if split == "test" else f"_{split}"
            suffix += "" if net_type == "vgg" else f"_{net_type}"
            with open(scene_dir + f"/results{suffix}.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + f"/per_view{suffix}.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--split', default="test", choices=["test", "val", "train"],
                         help="test = self-reenactment, val = novel-view synthesis")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--net_type', default="vgg", choices=["vgg", "alex", "squeeze"])
    args = parser.parse_args()
    evaluate(args.model_paths, args.split, args.batch_size, args.net_type)
