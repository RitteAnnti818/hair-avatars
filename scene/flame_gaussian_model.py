# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from vht.model.flame import FlameHead
from flame_model.flame import FlameHead

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, rotvec_to_unitquat, quat_product


class FlameGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, disable_flame_static_offset=False, not_finetune_flame_params=False, n_shape=300, n_expr=100,
                 enable_hair_strands=False, strand_json_path="", disable_strand_dynamic=False,
                 enable_motion_gate=False, motion_gate_percentile=90.0,
                 strand_temporal_mode="none", strand_dynamic_tip_power=1.0,
                 enable_strand_rotation=False, enable_rebinding=False):
        super().__init__(sh_degree)

        self.disable_flame_static_offset = disable_flame_static_offset
        self.not_finetune_flame_params = not_finetune_flame_params
        self.n_shape = n_shape
        self.n_expr = n_expr

        # HairAvatars: strand-chain soft-rigging (opt-in, defaults keep exact baseline behavior)
        # Delta_j(t) = static_j (time-invariant shape correction) + dynamic_j(t) (time-varying residual, ablatable)
        self.enable_hair_strands = enable_hair_strands
        self.strand_json_path = strand_json_path
        self.disable_strand_dynamic = disable_strand_dynamic
        self.face_strand_id = None
        self.face_chain_pos = None
        self.strand_chain_len = None
        self.num_strands = 0
        self.max_chain_len = 0
        self._strand_delta_static = None
        self._strand_delta_dynamic = None
        self.strand_delta_cumsum = None

        # HairAvatars: motion-gated dynamic term, g(m(t)) = clip(m(t)/m_ref, 0, 1), m(t) = finite-difference
        # pose velocity (rotation + neck_pose). Opt-in; when off, dynamic_j(t) is applied unconditionally
        # (previous behavior). m_ref is calibrated once per subject from its own training-sequence motion
        # distribution (fixed percentile rule, not tuned per-subject to maximize held-out metrics).
        self.enable_motion_gate = enable_motion_gate
        self.motion_gate_percentile = motion_gate_percentile
        self.strand_temporal_mode = strand_temporal_mode
        self.strand_dynamic_tip_power = strand_dynamic_tip_power
        self.strand_dynamic_active = True
        self.motion_gate_ref = None
        self.last_motion_gate = None  # most recent gate value, exposed for logging/debugging
        self.strand_temporal_gate = None
        self.strand_temporal_gate_l2 = None

        # HairAvatars: chain-propagated strand rotation. Delta_j(t) (position) has a rotation analogue:
        # per-link axis-angle rot_j, composed via quaternion chain Q_j = Q_{j-1} (x) exp(w_j * rot_j),
        # Q_0 = Identity (root stays rigid). Zero-initialized so exp(0) = identity -> exact no-op until
        # training moves it away from zero, same opt-in/safe-by-construction pattern as the position chain.
        self.enable_strand_rotation = enable_strand_rotation
        self._strand_rotation_static = None
        self.strand_rotation_cumprod = None

        # HairAvatars: periodic nearest-triangle rebinding (independent of the strand chain).
        # Diagnostic for whether GaussianAvatars' permanent triangle binding is itself the
        # bottleneck (cf. TeGA's critique of "greedy, potentially suboptimal" binding), tested
        # with a much lighter nearest-center reassignment instead of TeGA's continuous UVD+Jacobian
        # formulation.
        self.enable_rebinding = enable_rebinding
        self.hair_face_ids = None

        self.flame_model = FlameHead(
            n_shape,
            n_expr,
            add_teeth=True,
        ).cuda()
        self.flame_param = None
        self.flame_param_orig = None

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.flame_model.faces)).cuda()
            self.binding_counter = torch.ones(len(self.flame_model.faces), dtype=torch.int32).cuda()

            if self.enable_hair_strands:
                self._init_strand_topology()

            if self.enable_rebinding:
                self.hair_face_ids = self.flame_model.mask.f.hair.cuda()
                print(f"[HairAvatars] rebinding enabled: {len(self.hair_face_ids)} hair-region "
                      f"candidate triangles")

    def rebind_hair_gaussians(self):
        """Reassign each hair-region Gaussian to its nearest hair triangle (by current world
        position), instead of keeping it permanently bound to its initial triangle. Lightweight
        proxy for TeGA's continuous cross-triangle rebinding (arXiv:2505.05672), which requires a
        full UVD/Jacobian reparameterization -- here we just do periodic nearest-center reassignment
        restricted to the fixed hair-triangle set, to test whether permanent binding is itself the
        bottleneck, independent of the strand chain mechanism (enable_hair_strands can be off)."""
        if self.hair_face_ids is None or self.hair_face_ids.numel() == 0:
            return
        is_hair_gaussian = (self.binding[:, None] == self.hair_face_ids[None, :]).any(dim=1)
        if not is_hair_gaussian.any():
            return
        idx = is_hair_gaussian.nonzero(as_tuple=True)[0]

        with torch.no_grad():
            world_xyz = self.get_xyz[idx]  # (M, 3), current world position (includes any active offset)
            hair_centers = self.face_center[self.hair_face_ids]  # (Fh, 3)
            dists = torch.cdist(world_xyz, hair_centers)  # (M, Fh)
            nearest_fid = self.hair_face_ids[dists.argmin(dim=1)]  # (M,)

            old_fid = self.binding[idx]
            changed = nearest_fid != old_fid
            if not changed.any():
                return
            ci = idx[changed]
            new_fid = nearest_fid[changed]
            old_fid_changed = old_fid[changed]

            # Re-express the same world position in the new triangle's local frame, so there is
            # no visual jump at the moment of rebinding (rotation/scale local params are left
            # as-is -- a small approximation that the optimizer corrects over subsequent iterations,
            # acceptable for this periodic/infrequent diagnostic mechanism).
            R_new = self.face_orien_mat[new_fid]
            k_new = self.face_scaling[new_fid]
            T_new = self.face_center[new_fid]
            local_xyz = torch.bmm(
                R_new.transpose(1, 2), (world_xyz[changed] - T_new)[..., None]
            ).squeeze(-1) / k_new  # k_new already (K,1), broadcasts against (K,3)
            self._xyz.data[ci] = local_xyz

            self.binding_counter.scatter_add_(
                0, old_fid_changed, -torch.ones_like(old_fid_changed, dtype=torch.int32))
            self.binding_counter.scatter_add_(
                0, new_fid, torch.ones_like(new_fid, dtype=torch.int32))
            self.binding[ci] = new_fid

        print(f"[HairAvatars] rebound {changed.sum().item()}/{len(idx)} hair Gaussians to a "
              f"different triangle")

    def _init_strand_topology(self):
        """Load Phase-1 strand definitions (dev/hair_avatars/phase1_strands_k32.json) and build
        per-face lookup tables: which strand a triangle belongs to, and its root(0)->tip position."""
        import json
        with open(self.strand_json_path) as f:
            strands = json.load(f)

        num_faces = len(self.flame_model.faces)
        face_strand_id = torch.full((num_faces,), -1, dtype=torch.long)
        face_chain_pos = torch.zeros((num_faces,), dtype=torch.long)

        self.num_strands = len(strands)
        chain_lens = torch.zeros((self.num_strands,), dtype=torch.long)
        for sid_str, data in strands.items():
            sid = int(sid_str)
            fids = data["face_ids"]
            chain_lens[sid] = len(fids)
            for pos, fid in enumerate(fids):
                face_strand_id[fid] = sid
                face_chain_pos[fid] = pos

        self.max_chain_len = int(chain_lens.max().item())
        self.face_strand_id = face_strand_id.cuda()
        self.face_chain_pos = face_chain_pos.cuda()
        self.strand_chain_len = chain_lens.cuda()
        print(f"[HairAvatars] loaded {self.num_strands} strands (max chain length {self.max_chain_len}) "
              f"covering {(face_strand_id >= 0).sum().item()}/{num_faces} faces")

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.flame_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes

            self.num_timesteps = max(pose_meshes) + 1  # required by viewers
            num_verts = self.flame_model.v_template.shape[0]

            if self.enable_hair_strands and self.face_strand_id is not None:
                self._strand_delta_static = torch.zeros(self.num_strands, self.max_chain_len, 3).cuda()
                if not self.disable_strand_dynamic:
                    self._strand_delta_dynamic = torch.zeros(
                        self.num_strands, self.max_chain_len, self.num_timesteps, 3
                    ).cuda()
                if self.enable_strand_rotation:
                    self._strand_rotation_static = torch.zeros(self.num_strands, self.max_chain_len, 3).cuda()

            if not self.disable_flame_static_offset:
                static_offset = torch.from_numpy(meshes[0]['static_offset'])
                if static_offset.shape[0] != num_verts:
                    static_offset = torch.nn.functional.pad(static_offset, (0, 0, 0, num_verts - meshes[0]['static_offset'].shape[1]))
            else:
                static_offset = torch.zeros([num_verts, 3])

            T = self.num_timesteps

            self.flame_param = {
                'shape': torch.from_numpy(meshes[0]['shape']),
                'expr': torch.zeros([T, meshes[0]['expr'].shape[1]]),
                'rotation': torch.zeros([T, 3]),
                'neck_pose': torch.zeros([T, 3]),
                'jaw_pose': torch.zeros([T, 3]),
                'eyes_pose': torch.zeros([T, 6]),
                'translation': torch.zeros([T, 3]),
                'static_offset': static_offset,
                'dynamic_offset': torch.zeros([T, num_verts, 3]),
            }

            for i, mesh in pose_meshes.items():
                self.flame_param['expr'][i] = torch.from_numpy(mesh['expr'])
                self.flame_param['rotation'][i] = torch.from_numpy(mesh['rotation'])
                self.flame_param['neck_pose'][i] = torch.from_numpy(mesh['neck_pose'])
                self.flame_param['jaw_pose'][i] = torch.from_numpy(mesh['jaw_pose'])
                self.flame_param['eyes_pose'][i] = torch.from_numpy(mesh['eyes_pose'])
                self.flame_param['translation'][i] = torch.from_numpy(mesh['translation'])
                # self.flame_param['dynamic_offset'][i] = torch.from_numpy(mesh['dynamic_offset'])
            
            for k, v in self.flame_param.items():
                self.flame_param[k] = v.float().cuda()

            self.flame_param_orig = {k: v.clone() for k, v in self.flame_param.items()}

            if self.enable_motion_gate:
                pose = torch.cat([self.flame_param['rotation'], self.flame_param['neck_pose']], dim=1)  # (T, 6)
                m = (pose[1:] - pose[:-1]).norm(dim=-1)  # (T-1,), per-frame pose velocity
                self.motion_gate_ref = torch.quantile(m, self.motion_gate_percentile / 100.0).clamp(min=1e-6)
                print(f"[HairAvatars] motion gate enabled: m_ref (p{self.motion_gate_percentile:.0f}) = "
                      f"{self.motion_gate_ref.item():.4f}")

            if self.strand_temporal_mode != 'none' and self.enable_hair_strands and self._strand_delta_static is not None:
                gate_out = self.num_strands if self.strand_temporal_mode == 'strand_gate' else 1
                self.strand_temporal_gate = nn.Sequential(
                    nn.Linear(12, 32),
                    nn.ReLU(),
                    nn.Linear(32, gate_out),
                ).cuda()
                print(f"[HairAvatars] temporal mode enabled: {self.strand_temporal_mode} (gate_out={gate_out})")
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def update_mesh_by_param_dict(self, flame_param):
        if 'shape' in flame_param:
            shape = flame_param['shape']
        else:
            shape = self.flame_param['shape']

        if 'static_offset' in flame_param:
            static_offset = flame_param['static_offset']
        else:
            static_offset = self.flame_param['static_offset']

        verts, verts_cano = self.flame_model(
            shape[None, ...],
            flame_param['expr'].cuda(),
            flame_param['rotation'].cuda(),
            flame_param['neck'].cuda(),
            flame_param['jaw'].cuda(),
            flame_param['eyes'].cuda(),
            flame_param['translation'].cuda(),
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=static_offset,
        )
        self.update_mesh_properties(verts, verts_cano)

    def select_mesh_by_timestep(self, timestep, original=False):
        self.timestep = timestep
        flame_param = self.flame_param_orig if original and self.flame_param_orig != None else self.flame_param

        verts, verts_cano = self.flame_model(
            flame_param['shape'][None, ...],
            flame_param['expr'][[timestep]],
            flame_param['rotation'][[timestep]],
            flame_param['neck_pose'][[timestep]],
            flame_param['jaw_pose'][[timestep]],
            flame_param['eyes_pose'][[timestep]],
            flame_param['translation'][[timestep]],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            static_offset=flame_param['static_offset'],
            dynamic_offset=flame_param['dynamic_offset'][[timestep]],
        )
        self.update_mesh_properties(verts, verts_cano)

        if self.enable_hair_strands and self._strand_delta_static is not None:
            self._update_strand_delta_cumsum(timestep)
            if self.enable_strand_rotation and self._strand_rotation_static is not None:
                self._update_strand_rotation_cumprod()

    def _temporal_history_features(self, timestep):
        rot, neck = self.flame_param['rotation'], self.flame_param['neck_pose']
        pose_t = torch.cat([rot[timestep], neck[timestep]])
        if timestep == 0:
            vel = torch.zeros_like(pose_t)
            acc = torch.zeros_like(pose_t)
        elif timestep == 1:
            pose_prev = torch.cat([rot[timestep - 1], neck[timestep - 1]])
            vel = pose_t - pose_prev
            acc = torch.zeros_like(pose_t)
        else:
            pose_prev = torch.cat([rot[timestep - 1], neck[timestep - 1]])
            pose_prev2 = torch.cat([rot[timestep - 2], neck[timestep - 2]])
            vel = pose_t - pose_prev
            acc = pose_t - 2 * pose_prev + pose_prev2
        return torch.cat([vel, acc])

    def _compute_motion_gate(self, timestep):
        """Default gate: clip(||pose(t)-pose(t-1)|| / m_ref, 0, 1). Temporal modes can replace this
        with history-aware gates without changing the rest of the chain update."""
        if timestep == 0:
            self.last_motion_gate = 0.0
            return 0.0
        rot, neck = self.flame_param['rotation'], self.flame_param['neck_pose']
        pose_t = torch.cat([rot[timestep], neck[timestep]])
        pose_prev = torch.cat([rot[timestep - 1], neck[timestep - 1]])
        m = (pose_t - pose_prev).norm()
        gate = (m / self.motion_gate_ref).clamp(0, 1)
        self.last_motion_gate = gate.item()
        return gate

    def _compute_temporal_gate(self, timestep):
        if self.strand_temporal_gate is None:
            return None
        feat = self._temporal_history_features(timestep).to(self.strand_temporal_gate[0].weight.device)
        gate_raw = self.strand_temporal_gate(feat[None, :]).squeeze(0)
        if gate_raw.numel() == 1:
            gate = torch.sigmoid(gate_raw).squeeze()
        else:
            gate = torch.sigmoid(gate_raw)
        return gate

    def _update_strand_delta_cumsum(self, timestep):
        """Accumulate the per-link learned offset delta_j = static_j + dynamic_j(t) along each
        strand's chain. The temporal mode can modulate dynamic_j(t) using either the original motion
        gate or a learned pose-history gate."""
        delta_t = self._strand_delta_static
        if self._strand_delta_dynamic is not None:
            dynamic_t = self._strand_delta_dynamic[:, :, timestep, :]
            if not self.strand_dynamic_active:
                dynamic_t = torch.zeros_like(dynamic_t)
            if self.strand_dynamic_tip_power != 1.0:
                link_idx_dyn = torch.arange(self.max_chain_len, device=dynamic_t.device).float()[None, :]
                denom_dyn = (self.strand_chain_len.float() - 1).clamp(min=1)[:, None]
                dynamic_ramp = (link_idx_dyn / denom_dyn).clamp(0, 1).pow(self.strand_dynamic_tip_power)
                dynamic_t = dynamic_t * dynamic_ramp[..., None]
            if self.strand_temporal_mode == 'pose_gate' and self.strand_temporal_gate is not None:
                gate = self._compute_temporal_gate(timestep)
                if gate is not None:
                    dynamic_t = dynamic_t * gate
            elif self.strand_temporal_mode == 'strand_gate' and self.strand_temporal_gate is not None:
                gate = self._compute_temporal_gate(timestep)
                if gate is not None:
                    dynamic_t = dynamic_t * gate[:, None, None]
            elif self.enable_motion_gate and self.motion_gate_ref is not None:
                dynamic_t = dynamic_t * self._compute_motion_gate(timestep)
            delta_t = delta_t + dynamic_t
        # (num_strands, max_chain_len, 3)
        link_idx = torch.arange(self.max_chain_len, device=delta_t.device).float()[None, :]  # (1, L)
        denom = (self.strand_chain_len.float() - 1).clamp(min=1)[:, None]  # (num_strands, 1)
        w = (link_idx / denom).clamp(0, 1)  # (num_strands, L), root->0, tip->1
        self.strand_delta_cumsum = torch.cumsum(delta_t * w[..., None], dim=1)  # (num_strands, max_chain_len, 3)

    def _update_strand_rotation_cumprod(self):
        """Rotation analogue of _update_strand_delta_cumsum: chain-compose per-link axis-angle rotations
        via quaternion product instead of summing (rotations don't add). Q_j = Q_{j-1} (x) exp(w_j * rot_j),
        Q_0 = Identity (root stays rigid). Same root-fixed, tip-increasing ramp as the position chain.
        Time-invariant (no dynamic term for rotation yet), so this only needs the static axis-angle."""
        aa = self._strand_rotation_static  # (num_strands, max_chain_len, 3) axis-angle, static
        link_idx = torch.arange(self.max_chain_len, device=aa.device).float()[None, :]  # (1, L)
        denom = (self.strand_chain_len.float() - 1).clamp(min=1)[:, None]  # (num_strands, 1)
        w = (link_idx / denom).clamp(0, 1)  # (num_strands, L), root->0, tip->1
        # scaling the axis-angle vector by w_j before exponentiating == taking that single-axis
        # rotation to the power w_j (exact for axis-angle, since it's the Lie algebra element)
        scaled_aa = aa * w[..., None]  # (num_strands, max_chain_len, 3)
        link_quats = rotvec_to_unitquat(scaled_aa)  # (num_strands, max_chain_len, 4) xyzw; w_j=0 -> identity

        cum = [link_quats[:, 0]]  # j=0: w_0=0 -> identity quat, root stays exactly rigid
        for j in range(1, self.max_chain_len):
            cum.append(quat_product(cum[-1], link_quats[:, j]))
        self.strand_rotation_cumprod = torch.stack(cum, dim=1)  # (num_strands, max_chain_len, 4) xyzw

    def compute_strand_coherence_loss(self, threshold):
        """Strand-Coherence Regularization: penalize ||Delta_j - Delta_{j-1}|| beyond `threshold`
        between adjacent links on the same strand chain. Unlike the static/dynamic magnitude
        losses (which only bound how far a single link may sit from the rigid mesh position),
        this bounds how much two *neighboring* Gaussians on a strand may move relative to each
        other, preventing zig-zag/snapping artifacts under extreme poses regardless of whether
        each individual link stays under its own magnitude threshold."""
        cumsum = self.strand_delta_cumsum  # (num_strands, max_chain_len, 3)
        if cumsum.shape[1] < 2:
            return cumsum.new_zeros(())
        diff = cumsum[:, 1:, :] - cumsum[:, :-1, :]  # (num_strands, max_chain_len-1, 3)
        link_idx = torch.arange(diff.shape[1], device=diff.device)[None, :]  # (1, L-1)
        valid = link_idx < (self.strand_chain_len[:, None] - 1)  # (num_strands, L-1): only real (non-padding) transitions
        penalty = F.relu(diff.norm(dim=-1) - threshold)[valid]
        if penalty.numel() == 0:
            return cumsum.new_zeros(())
        return penalty.mean()

    def compute_strand_temporal_smoothness_loss(self, threshold=0.0):
        """A: temporal smoothness on the learned strand offsets. Penalizes frame-to-frame change
        in the accumulated chain offset, either with a hard threshold (ReLU) or as a plain L2 if
        threshold is zero."""
        if self._strand_delta_dynamic is None or self._strand_delta_dynamic.shape[2] < 2:
            return self._strand_delta_static.new_zeros(())
        diff = self._strand_delta_dynamic[:, :, 1:, :] - self._strand_delta_dynamic[:, :, :-1, :]
        if threshold > 0:
            penalty = F.relu(diff.norm(dim=-1) - threshold)
            return penalty.mean()
        return diff.norm(dim=-1).mean()

    def update_mesh_properties(self, verts, verts_cano):
        faces = self.flame_model.faces
        triangles = verts[:, faces]

        # position
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), return_scale=True)
        # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
        self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma

        # for mesh rendering
        self.verts = verts
        self.faces = faces

        # for mesh regularization
        self.verts_cano = verts_cano
    
    def compute_dynamic_offset_loss(self):
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):
        super().training_setup(training_args)

        if self.enable_hair_strands and self._strand_delta_static is not None:
            self._strand_delta_static.requires_grad_(True)
            param_strand_static = {'params': [self._strand_delta_static], 'lr': training_args.strand_delta_lr, "name": "strand_delta_static"}
            self.optimizer.add_param_group(param_strand_static)
            if self._strand_delta_dynamic is not None:
                self._strand_delta_dynamic.requires_grad_(True)
                dynamic_lr = training_args.strand_dynamic_lr if training_args.strand_dynamic_lr > 0 else training_args.strand_delta_lr
                param_strand_dynamic = {'params': [self._strand_delta_dynamic], 'lr': dynamic_lr, "name": "strand_delta_dynamic"}
                self.optimizer.add_param_group(param_strand_dynamic)

            if self.strand_temporal_gate is not None:
                for p in self.strand_temporal_gate.parameters():
                    p.requires_grad_(True)
                param_temporal_gate = {'params': list(self.strand_temporal_gate.parameters()),
                                       'lr': training_args.strand_delta_lr, "name": "strand_temporal_gate"}
                self.optimizer.add_param_group(param_temporal_gate)

            if self.enable_strand_rotation and self._strand_rotation_static is not None:
                self._strand_rotation_static.requires_grad_(True)
                param_strand_rot = {'params': [self._strand_rotation_static], 'lr': training_args.strand_delta_lr,
                                     "name": "strand_rotation"}
                self.optimizer.add_param_group(param_strand_rot)

        if self.not_finetune_flame_params:
            return

        # # shape
        # self.flame_param['shape'].requires_grad = True
        # param_shape = {'params': [self.flame_param['shape']], 'lr': 1e-5, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # pose
        self.flame_param['rotation'].requires_grad = True
        self.flame_param['neck_pose'].requires_grad = True
        self.flame_param['jaw_pose'].requires_grad = True
        self.flame_param['eyes_pose'].requires_grad = True
        params = [
            self.flame_param['rotation'],
            self.flame_param['neck_pose'],
            self.flame_param['jaw_pose'],
            self.flame_param['eyes_pose'],
        ]
        param_pose = {'params': params, 'lr': training_args.flame_pose_lr, "name": "pose"}
        self.optimizer.add_param_group(param_pose)

        # translation
        self.flame_param['translation'].requires_grad = True
        param_trans = {'params': [self.flame_param['translation']], 'lr': training_args.flame_trans_lr, "name": "trans"}
        self.optimizer.add_param_group(param_trans)
        
        # expression
        self.flame_param['expr'].requires_grad = True
        param_expr = {'params': [self.flame_param['expr']], 'lr': training_args.flame_expr_lr, "name": "expr"}
        self.optimizer.add_param_group(param_expr)

        # # static_offset
        # self.flame_param['static_offset'].requires_grad = True
        # param_static_offset = {'params': [self.flame_param['static_offset']], 'lr': 1e-6, "name": "static_offset"}
        # self.optimizer.add_param_group(param_static_offset)

        # # dynamic_offset
        # self.flame_param['dynamic_offset'].requires_grad = True
        # param_dynamic_offset = {'params': [self.flame_param['dynamic_offset']], 'lr': 1.6e-6, "name": "dynamic_offset"}
        # self.optimizer.add_param_group(param_dynamic_offset)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "flame_param.npz"
        flame_param = {k: v.cpu().numpy() for k, v in self.flame_param.items()}
        np.savez(str(npz_path), **flame_param)

        if self.enable_hair_strands and self._strand_delta_static is not None:
            strand_path = Path(path).parent / "strand_delta.npz"
            strand_data = {"static": self._strand_delta_static.detach().cpu().numpy()}
            if self._strand_delta_dynamic is not None:
                strand_data["dynamic"] = self._strand_delta_dynamic.detach().cpu().numpy()
            if self.enable_strand_rotation and self._strand_rotation_static is not None:
                strand_data["rotation_static"] = self._strand_rotation_static.detach().cpu().numpy()
            np.savez(str(strand_path), **strand_data)

            # BUGFIX: the learned temporal gate (pose_gate/strand_gate) was never persisted, so
            # render.py's freshly-reinitialized (random, untrained) gate silently replaced the
            # trained one at eval time. Save its weights alongside the strand deltas.
            if self.strand_temporal_gate is not None:
                gate_path = Path(path).parent / "temporal_gate.pt"
                torch.save(self.strand_temporal_gate.state_dict(), str(gate_path))

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "flame_param.npz"
            flame_param = np.load(str(npz_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items()}

            self.flame_param = flame_param
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers

        if self.enable_hair_strands and self.face_strand_id is not None:
            strand_path = Path(path).parent / "strand_delta.npz"
            if strand_path.exists():
                strand_data = np.load(str(strand_path))
                self._strand_delta_static = torch.from_numpy(strand_data["static"]).cuda()
                if "dynamic" in strand_data:
                    self._strand_delta_dynamic = torch.from_numpy(strand_data["dynamic"]).cuda()
                if self.enable_strand_rotation and "rotation_static" in strand_data:
                    self._strand_rotation_static = torch.from_numpy(strand_data["rotation_static"]).cuda()
                print(f"[HairAvatars] restored strand_delta from {strand_path}")
            else:
                print(f"[HairAvatars] WARNING: {strand_path} not found — strand offsets reset to zero")

            # BUGFIX (see save_ply): restore the trained temporal gate weights instead of leaving
            # the randomly-initialized one created by load_meshes().
            if self.strand_temporal_gate is not None:
                gate_path = Path(path).parent / "temporal_gate.pt"
                if gate_path.exists():
                    self.strand_temporal_gate.load_state_dict(torch.load(str(gate_path), map_location="cuda"))
                    print(f"[HairAvatars] restored temporal_gate from {gate_path}")
                else:
                    print(f"[HairAvatars] WARNING: {gate_path} not found — temporal gate left randomly initialized")
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            flame_param = np.load(str(motion_path))
            flame_param = {k: torch.from_numpy(v).cuda() for k, v in flame_param.items() if v.dtype == np.float32}

            self.flame_param = {
                # keep the static parameters
                'shape': self.flame_param['shape'],
                'static_offset': self.flame_param['static_offset'],
                # update the dynamic parameters
                'translation': flame_param['translation'],
                'rotation': flame_param['rotation'],
                'neck_pose': flame_param['neck_pose'],
                'jaw_pose': flame_param['jaw_pose'],
                'eyes_pose': flame_param['eyes_pose'],
                'expr': flame_param['expr'],
                'dynamic_offset': flame_param['dynamic_offset'],
            }
            self.num_timesteps = self.flame_param['expr'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]
