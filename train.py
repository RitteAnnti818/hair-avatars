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

import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, num_workers=8):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params,
                                        enable_hair_strands=dataset.enable_hair_strands, strand_json_path=dataset.strand_json_path,
                                        disable_strand_dynamic=dataset.disable_strand_dynamic,
                                        enable_motion_gate=dataset.enable_motion_gate, motion_gate_percentile=dataset.motion_gate_percentile,
                                        enable_strand_rotation=dataset.enable_strand_rotation,
                                        enable_rebinding=dataset.enable_rebinding,
                                        enable_inextensible_chain=dataset.enable_inextensible_chain,
                                        inextensible_static_cap=dataset.inextensible_static_cap,
                                        inextensible_dynamic_cap=dataset.inextensible_dynamic_cap)
        mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loader_camera_train = DataLoader(scene.getTrainCameras(), batch_size=None, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    iter_camera_train = iter(loader_camera_train)
    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # receive data
                net_image = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, use_original_mesh = network_gui.receive()
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, custom_cam)

                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        mesh_opacity = msg['mesh_opacity']
                        if net_image is None:
                            net_image = rgb_mesh
                        else:
                            net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))

                    # send data
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Gradient accumulation: render+backward `grad_accum_steps` single-view samples before the
        # optimizer step below, since the rasterizer/mesh state (select_mesh_by_timestep) only supports
        # one view/timestep per render call. grad_accum_steps=1 (default) reproduces the exact previous
        # per-iteration behavior. `iteration` still counts optimizer steps, not individual renders, so
        # all iteration-gated schedules (LR, SH degree, densification, checkpointing) are unaffected.
        grad_accum_steps = max(1, opt.gradient_accumulation_steps)
        for _sub_step in range(grad_accum_steps):
            try:
                viewpoint_cam = next(iter_camera_train)
            except StopIteration:
                iter_camera_train = iter(loader_camera_train)
                viewpoint_cam = next(iter_camera_train)

            if gaussians.binding != None:
                gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

            # Render
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()

            losses = {}
            losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
            losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim

            if gaussians.binding != None:
                # Guard against views where zero Gaussians pass the visibility filter (can happen
                # early in training with a sparse initial point cloud + an extreme camera angle,
                # e.g. profile/back-of-head views in HAIR sequences). `tensor[empty_mask].mean()`
                # is NaN in PyTorch, which then poisons every parameter via backward() and never
                # recovers. Skip these visibility-filtered regularizers for that iteration instead.
                n_visible = visibility_filter.sum()

                if opt.metric_xyz:
                    if n_visible > 0:
                        losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
                    else:
                        losses['xyz'] = torch.zeros((), device=gaussians._xyz.device)
                else:
                    # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                    if n_visible > 0:
                        losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz
                    else:
                        losses['xyz'] = torch.zeros((), device=gaussians._xyz.device)

                if opt.lambda_scale != 0:
                    if opt.metric_scale:
                        if n_visible > 0:
                            losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                        else:
                            losses['scale'] = torch.zeros((), device=gaussians._xyz.device)
                    else:
                        # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                        if n_visible > 0:
                            losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                        else:
                            losses['scale'] = torch.zeros((), device=gaussians._xyz.device)

                if opt.lambda_dynamic_offset != 0:
                    losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

                if opt.lambda_dynamic_offset_std != 0:
                    ti = viewpoint_cam.timestep
                    t_indices =[ti]
                    if ti > 0:
                        t_indices.append(ti-1)
                    if ti < gaussians.num_timesteps - 1:
                        t_indices.append(ti+1)
                    losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std

                if opt.lambda_laplacian != 0:
                    losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian

                # HairAvatars: regularize the strand offset Delta_j(t) = static_j + dynamic_j(t)
                if gaussians.enable_hair_strands and gaussians.strand_delta_cumsum is not None:
                    # Under enable_inextensible_chain (v2), ||static_j||/||dynamic_j(t)|| < cap always
                    # by construction (sigmoid-gated magnitude) -- the ReLU-threshold penalty below has
                    # no job left to do (it would always evaluate to ~0 if cap<=threshold, and
                    # gaussians._strand_delta_static/_dynamic don't even exist as tensors in this mode),
                    # so skip it entirely rather than keep a redundant/dead loss term.
                    if not gaussians.enable_inextensible_chain and gaussians._strand_delta_static is not None:
                        if opt.lambda_strand_static != 0:
                            losses['strand_static'] = F.relu(
                                gaussians._strand_delta_static.norm(dim=-1) - opt.threshold_strand_static
                            ).mean() * opt.lambda_strand_static
                        if opt.lambda_strand_dynamic != 0 and gaussians._strand_delta_dynamic is not None:
                            # sparsity on the time-varying residual: prefer the static explanation (Occam's razor),
                            # only let dynamic_j(t) grow when the static offset alone cannot explain the data
                            losses['strand_dynamic'] = F.relu(
                                gaussians._strand_delta_dynamic.norm(dim=-1) - opt.threshold_strand_dynamic
                            ).mean() * opt.lambda_strand_dynamic
                        if opt.lambda_strand_temporal_smooth != 0 and gaussians._strand_delta_dynamic is not None:
                            # A-1: does *any* temporal coupling on the dynamic residual help, given it
                            # currently has none (epsilon(t) and epsilon(t-1) are fully independent)?
                            # Cheap diagnostic before investing in real motion supervision (optical flow).
                            losses['strand_temporal_smooth'] = gaussians.compute_strand_temporal_smoothness_loss() \
                                * opt.lambda_strand_temporal_smooth
                    if opt.lambda_strand_coherence != 0:
                        # C2: Strand-Coherence Regularization — relative rigidity between adjacent chain links
                        losses['strand_coherence'] = gaussians.compute_strand_coherence_loss(
                            opt.threshold_strand_coherence
                        ) * opt.lambda_strand_coherence
                    if gaussians.enable_strand_rotation and gaussians._strand_rotation_static is not None \
                            and opt.lambda_strand_rotation != 0:
                        # chain-propagated rotation: penalize per-link axis-angle magnitude beyond threshold
                        losses['strand_rotation'] = F.relu(
                            gaussians._strand_rotation_static.norm(dim=-1) - opt.threshold_strand_rotation
                        ).mean() * opt.lambda_strand_rotation

            losses['total'] = sum([v for k, v in losses.items()])
            # scale by 1/grad_accum_steps so accumulated .grad matches the mean-of-batch gradient
            # (PyTorch accumulates gradients additively across backward() calls by default)
            (losses['total'] / grad_accum_steps).backward()

            # Densification stats: accumulate per sub-step, same as each view would contribute
            # independently under true batching.
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"Loss": f"{ema_loss_for_log:.{7}f}"}
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"
                if 'strand_static' in losses:
                    postfix["strand_s"] = f"{losses['strand_static']:.{7}f}"
                if 'strand_dynamic' in losses:
                    postfix["strand_d"] = f"{losses['strand_dynamic']:.{7}f}"
                if 'strand_coherence' in losses:
                    postfix["strand_c"] = f"{losses['strand_coherence']:.{7}f}"
                if 'strand_rotation' in losses:
                    postfix["strand_r"] = f"{losses['strand_rotation']:.{7}f}"
                if 'strand_temporal_smooth' in losses:
                    postfix["strand_ts"] = f"{losses['strand_temporal_smooth']:.{7}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification (stats already accumulated per sub-step in the render loop above)
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # HairAvatars: periodic nearest-triangle rebinding for hair-region Gaussians (opt-in)
            if getattr(gaussians, 'enable_rebinding', False) and iteration % opt.rebinding_interval == 0:
                gaussians.rebind_hair_gaussians()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras()},
            {'name': 'test', 'cameras' : scene.getTestCameras()},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 10
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    image_cache.append(image)
                    gt_image_cache.append(gt_image)

                    if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                        batch_img = torch.stack(image_cache, dim=0)
                        batch_gt_img = torch.stack(gt_image_cache, dim=0)
                        lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                        image_cache = []
                        gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--num_workers", type=int, default=8,
                         help="DataLoader worker processes for the training loop. Lower this (e.g. 2-3) "
                              "when running multiple training jobs in parallel on a shared machine to "
                              "reduce total CPU footprint.")
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.num_workers)

    # All done
    print("\nTraining complete.")
