# Next-Steps Plan — Implementation & Experiments Only (2026-08-19)

Scope: B (inextensible chain) and A (optical-flow / temporal supervision), code-level steps + experiment
matrix. C (dynamic kNN) parked — not in scope. Paper-writing/framing tasks intentionally excluded here.

Confirmed by code read: `get_xyz` (`scene/gaussian_model.py:156-184`) consumes `strand_delta_cumsum`
(position); `get_rotation` (`scene/gaussian_model.py:126-153`) consumes `strand_rotation_cumprod`
(splat orientation only). These are fully independent — the already-tried "chain-propagated rotation"
ablation only touched `get_rotation`. B below targets the `get_xyz`/`strand_delta_cumsum` path and does
not overlap with that prior experiment.

---

## B. Inextensible (arc-length-preserving) position chain — **IMPLEMENTED (2026-08-19), design revised**

**Design revision, superseding everything originally drafted below (kept for the record, not the
current design)**: the DER-style "rest-segment + rotation" approach originally planned here does not
fit this codebase. `strand_delta_cumsum` is a *small additive correction* on top of an already-rigid
per-triangle binding (`get_xyz`, `scene/gaussian_model.py:156-184`), not a from-scratch physical
centerline — importing actual canonical-mesh face-to-face rest lengths would (a) double-count geometry
already provided by the rigid binding (bend=0 would reproduce a large nonzero chain shape instead of
exactly zero), and (b) break the zero-init-safety property every other opt-in mechanism in this file
relies on. Caught during implementation, before any of that was written.

**What was actually implemented instead — hard norm-clamp, structurally equivalent goal, correctly
scoped to what this codebase's offset represents:**
$$\delta_{s,j}(t) = \delta_{s,j}^{raw}(t)\cdot\min\!\Big(1,\ \frac{\tau}{\lVert\delta_{s,j}^{raw}(t)\rVert}\Big)$$
applied to the static and dynamic components separately (their own caps), before the existing cumsum.
No-op below the cap (zero-init untouched); above the cap, rescaled to land exactly on it, direction
preserved. Since $\Delta_j - \Delta_{j-1} = w_j\delta_j$, this also hard-bounds the adjacent-link
separation C2 only discourages softly — targets the likely actual comet-tail mechanism (a single/few
near-tip links, where $w_j\to1$, developing an unbounded offset because the ReLU penalty can be
outweighed by photometric gradient) structurally rather than via penalty weight.

**Landed changes:**
- `arguments/__init__.py`: `enable_inextensible_chain` (bool), `inextensible_static_cap` (default 0.3,
  matches `threshold_strand_static`), `inextensible_dynamic_cap` (default 0.05, matches
  `threshold_strand_dynamic`) — all in `ModelParams`.
- `scene/flame_gaussian_model.py`: constructor accepts the three new args and stores them; new
  `_clamp_norm` static method; `_update_strand_delta_cumsum` clamps `_strand_delta_static` and
  `_strand_delta_dynamic` separately (before the motion gate) when the flag is set.
- `train.py`, `render.py`: wired through (render.py via `getattr(dataset, ..., default)`, matching the
  existing backward-compat pattern for checkpoints saved before hair-strand support existed).

**Verified (2026-08-19):**
- Unit test of `_clamp_norm` in isolation: below-cap exact no-op, above-cap rescales to exactly the cap
  with direction preserved, zero stays zero, boundary continuous — all passed.
- Real `FlameGaussianModel` instantiated on GPU 7 with `enable_hair_strands=True,
  enable_inextensible_chain=True` — strand topology loads (32 strands, max chain length 34), and
  deliberately oversized fake deltas (~5.0 magnitude vs. 0.3 cap) get clamped to exactly the cap through
  `_update_strand_delta_cumsum`. No actual training run yet (B-2 below).

**Built-in sanity signature for the first real training run**: with
`inextensible_static_cap == threshold_strand_static`, `losses['strand_static']` (train.py's ReLU
penalty) should be provably ~0 for the entire run, since the clamp already guarantees
$\|\delta\|\le\tau$. If it's not ~0, the clamp isn't wired to the loss computation's tensors correctly —
check this first before trusting any B-2 metric.

---

<details>
<summary>Superseded original draft (DER rest-segment + composed rotation) — kept for reference only, not
the implemented design</summary>

### B-0. Rest-segment precompute — extend `_init_strand_topology` (`scene/flame_gaussian_model.py:143-169`)

`phase1_strands_k32.json`'s `face_ids` list (root→tip, already loaded at L158) gives everything needed —
no new artifact file required. Add right after the existing `chain_lens`/`face_chain_pos` loop:

```python
# canonical (template-pose) face centroid per face, for rest-segment vectors
verts_cano = self.flame_model.v_template  # (num_verts, 3)
faces = self.flame_model.faces            # (num_faces, 3)
face_centroids = verts_cano[faces].mean(dim=1)  # (num_faces, 3)

strand_rest_seg = torch.zeros(self.num_strands, self.max_chain_len, 3)
strand_rest_len = torch.zeros(self.num_strands, self.max_chain_len)
for sid_str, data in strands.items():
    sid = int(sid_str)
    fids = torch.tensor(data["face_ids"])
    centroids = face_centroids[fids]                    # (n_s, 3)
    seg = centroids[1:] - centroids[:-1]                 # (n_s-1, 3), link j uses centroids[j]-centroids[j-1]
    strand_rest_seg[sid, 1:len(fids)] = seg
    strand_rest_len[sid, 1:len(fids)] = seg.norm(dim=-1)

self.strand_rest_seg = strand_rest_seg.cuda()   # (K, max_chain_len, 3), index 0 unused (root)
self.strand_rest_len = strand_rest_len.cuda()   # (K, max_chain_len)
```

Sanity check before moving on: print `strand_rest_len` min/max/mean — should roughly match the
magnitude scale already implied by `threshold_strand_static=0.3` (triangle-relative units); if it's off
by an order of magnitude, the `k_{f(i)}` scaling convention needs revisiting (same bug class as the
"missing `k_f(i)` factor" note already in `method_equations.md` L157-160).

### B-1. Bend-only parameterization — new sibling method next to `_update_strand_delta_cumsum` (L301-317)

**Design correction (2026-08-19, superseding the first draft below-the-fold in git history): the
rotation applied to each rest segment must be the *cumulative* (chain-composed) rotation up to that
link, not an independent per-link rotation.** An independent `R_j` applied only to `e_rest_j` preserves
length but decouples each link's direction from its upstream neighbors — bending link 5 would not
reorient links 6, 7, 8..., which is not how a rod/strand actually bends and reopens a zig-zag failure
mode (the very thing C2 exists to prevent). The correct centerline update composes rotations along the
chain — mathematically identical to what `_update_strand_rotation_cumprod` (L319-336) *already computes*
as `strand_rotation_cumprod`, just currently wired only into `get_rotation` (splat orientation,
`scene/gaussian_model.py:143-151`), never into position. **B-1 is therefore mostly a rewiring of an
existing computation, not a new mechanism**: reuse `strand_rotation_cumprod`'s cumulative quaternion
$Q_j(t) = Q_{j-1}(t)\otimes\exp(w_j\cdot b_{s,j}(t))$ and apply it to the rest segment to get position,
*in addition to* its existing use for orientation — so a Gaussian's position and its visual orientation
now derive from the same rigid rotation at each link, instead of the previous (net-negative) experiment
where they were independent.

Replace the free-vector param tensors with axis-angle **bend** tensors of the same shape (drop-in
shape-compatible with existing `_strand_delta_static`/`_strand_delta_dynamic` allocation at L180-185, so
`__init__`/`load_meshes` allocation code barely changes — just rename `_strand_delta_*` →
`_strand_bend_*` when `enable_inextensible_chain` is set). This subsumes `enable_strand_rotation`'s
separate static rotation tensor — under this design there is only one bend field, driving both position
and orientation, not two independent ones:

```python
def _update_strand_delta_cumsum_inextensible(self, timestep):
    """Bend-only reparameterization: e_j(t) = Q_j(t) . e_rest_j, ||e_j(t)|| = l_j exactly (rotation
    preserves norm) -- replaces the free-vector delta_j with a hard-constrained-length alternative.
    Q_j is the *cumulative* chain rotation (same composition as _update_strand_rotation_cumprod), not an
    independent per-link rotation -- required so a bend at link j reorients all downstream links j+1..,
    matching how a rod/strand actually bends."""
    bend_t = self._strand_bend_static
    if self._strand_bend_dynamic is not None:
        dynamic_t = self._strand_bend_dynamic[:, :, timestep, :]
        if self.enable_motion_gate and self.motion_gate_ref is not None:
            dynamic_t = dynamic_t * self._compute_motion_gate(timestep)
        bend_t = bend_t + dynamic_t
    link_idx = torch.arange(self.max_chain_len, device=bend_t.device).float()[None, :]
    denom = (self.strand_chain_len.float() - 1).clamp(min=1)[:, None]
    w = (link_idx / denom).clamp(0, 1)                        # same root-fixed ramp as before
    scaled_aa = bend_t * w[..., None]                          # (K, L, 3)
    link_quats = rotvec_to_unitquat(scaled_aa)                  # (K, L, 4) xyzw, identity at root

    # cumulative (chain-composed) rotation -- identical loop to _update_strand_rotation_cumprod L333-336
    cum = [link_quats[:, 0]]
    for j in range(1, self.max_chain_len):
        cum.append(quat_product(cum[-1], link_quats[:, j]))
    Q = torch.stack(cum, dim=1)                                 # (K, L, 4) xyzw, cumulative

    self.strand_rotation_cumprod = Q                            # also drives get_rotation, unchanged consumer
    e = quat_rotate(Q, self.strand_rest_seg)                    # (K, L, 3), norm-preserving by construction
    self.strand_delta_cumsum = torch.cumsum(e, dim=1)           # same downstream shape/consumer as before
```

(`quat_rotate` needs to exist or be added next to `rotvec_to_unitquat`/`quat_product` — check
`utils/general_utils.py` or wherever those two already live; if missing, `R @ v` via
`roma.unitquat_to_rotmat` composed with the quat is an equally fine one-line substitute.)

Note the remaining, deliberately-open degree of freedom: this only constrains **adjacent-link
separation** to exactly $l_j$ — it does not cap how far the whole chain bends, so a strand can still
sweep through a large 3D motion if every link bends consistently in one direction (this is the wanted
dynamic behavior). What's structurally excluded is specifically the comet-tail failure mode (neighbors
pulling apart beyond rest length), not legitimate large-amplitude bending.

New flag in `arguments/__init__.py` next to L66-69:
```python
self.enable_inextensible_chain = False  # opt-in: bend-only rest-segment reparam, replaces free-vector
                                          # position offset; requires enable_hair_strands
```
Wire the branch in `load_meshes`/`update_mesh_properties` (wherever `_update_strand_delta_cumsum` is
currently called at L282) to call the inextensible variant instead when the flag is set. **Do not modify
the existing free-vector path** — keep both live behind the flag so the static-only baseline
(+0.281dB) stays bit-for-bit reproducible.

$\mathcal{L}_{static}$/$\mathcal{L}_{dynamic}$ magnitude penalties (currently on $\|\delta_{s,j}\|$,
`method_equations.md` L180-184) become penalties on bend angle $\|b_{s,j}\|$ (radians) — new thresholds
needed, not reused as-is. Start from `threshold_strand_rotation=0.1` (already exists for the rotation
ablation, same units — axis-angle radians) as the initial guess, sweep on 306 only if B-2a looks
promising.

</details>

### B-2. Experiment matrix (reuses `precise_hair_crop_eval.py` / `precise_hair_crop_multi.py`, same
9-subject/FREE protocol, same command pattern as `run_remaining_motiongate.sh`) — **flag name/command
pattern below is accurate for the implemented hard-clamp design as well, no changes needed there**

| Step | Command (pattern) | Subjects | Compare against | Gate to proceed |
|---|---|---|---|---|
| B-2a sanity | `train.py ... --enable_hair_strands --enable_inextensible_chain` (dynamic off) | 306, 302 | free-vector static-only (+0.281dB, +0.014dB) | no regression → continue |
| B-2b main | same + `--enable_motion_gate`-equivalent unconditional dynamic on ($g\equiv1$) | all 9 | free-vector naive dynamic (−0.004dB avg, harmful) | ≥6/9 non-negative AND 9-subj avg ≥0 → promote; else → report as data point, deprioritize further B work |
| B-2c optional | + `--enable_motion_gate --motion_gate_percentile 90.0` | all 9 | free-vector motion-gated (−0.080dB avg) | only run if B-2b passes |

Exact commands: copy `run_remaining_motiongate.sh`'s `train.py` invocation verbatim, add
`--enable_inextensible_chain`, change `-m output/...` naming (e.g. `output/inext_${subj}`), keep
`CUDA_VISIBLE_DEVICES=7` (only GPU to use). Eval: `precise_hair_crop_multi.py` with
`${subj}_inext`-style run names, same as existing `_motiongate`/`_strandrot` suffix convention.

Effort: B-0+B-1 code ~0.5–1 day. B-2a ~1 short training run × 2 subjects (use existing per-subject
wall-clock from motion-gate runs as reference). B-2b full 9-subject ~1–2 days compute (single GPU 7,
sequential — check whether prior sweeps were run sequentially or if multiple were interleaved; if
sequential, budget accordingly).

---

## A. Optical-flow / temporal supervision

### A-0. Flow-quality check (no training-code changes)

- Subjects: 264 (highest peakiness, best motion-gate result), 304 (2nd-highest peakiness, worst
  motion-gate result — unexplained anomaly), 218 (low peakiness, 2nd-best result). Reuse peakiness
  numbers already computed for the motion-gate writeup.
- Run RAFT (or equivalent) on consecutive GT frames, restricted to the hair mask already used by
  `precise_hair_crop_eval.py`.
- Check 1 (qualitative): overlay flow vectors on hair crops for frames already flagged problematic
  (comet-tail smear frames, 304's regression frames) — coherent along strand direction vs. noisy.
- Check 2 (quantitative): forward-backward flow consistency error inside hair mask vs. rest of
  face/body region — large gap ⇒ RAFT unreliable in the region that matters. This also gives a cheap
  side-check on the still-open 304 hypothesis (FLAME-tracking noise, `method_equations.md` L78-79):
  if 304's hair-region flow error is anomalously high relative to the other 8 subjects, that's
  corroborating (not conclusive) evidence.
- **Gate**: proceed to A-2 only if flow is structured/low-error in the hair region.

### A-1. Temporal-smoothness-only ablation (no RAFT, no dataloader change)

$\epsilon_{s,j}(t)$ (or `_strand_bend_dynamic` if B lands first) is a directly-learned parameter table
indexed by $t$ — the smoothness term can be computed from the parameter tensor directly at any training
step, independent of which frame the batch sampled that step:

```python
def compute_strand_temporal_smoothness_loss(self):
    eps = self._strand_delta_dynamic  # (K, L, T, 3)  [or _strand_bend_dynamic under B]
    diff = eps[:, :, 1:, :] - eps[:, :, :-1, :]
    valid = ... # same chain-length masking pattern as compute_strand_coherence_loss (L338-354)
    return diff.norm(dim=-1)[valid].pow(2).mean()
```

New flag/weight `lambda_strand_temporal_smooth`, added to the total objective alongside
`lambda_strand_coherence`. Sweep on 2-3 subjects (e.g. 306, 264, 304) first — cheapest possible signal on
whether *any* temporal coupling helps, independent of A-0's outcome.

### A-2. Full flow-consistency loss (gated on A-0 passing + remaining time)

1. Offline: cache RAFT flow per subject/frame/camera within hair mask (`.npy`, same crop geometry as
   `precise_hair_crop_eval.py` already computes).
2. Add a low-frequency auxiliary sampling stream (e.g. every $N$-th iteration, sample a random adjacent
   $(t, t{+}1)$ pair for one camera) **on top of** the existing `shuffle=True` single-frame loop in
   `train.py:60` — don't replace it, just add a second sampling path so the main photometric training is
   untouched if this experiment needs to be aborted.
3. Loss: project $\Delta_{s,j}(t{+}1) - \Delta_{s,j}(t)$ (world-space, via `face_orien_mat`/`face_center`
   at both timesteps) into image space with the camera's projection matrix; compare to cached flow,
   masked to hair region, optionally weighted by the A-0 forward-backward confidence map.
4. Multi-view: NeRSemble has 16 views — try averaging the projected-flow loss over 2+ cameras per pair
   before committing to single-view, for robustness against any one camera's flow being bad.
5. Tune $\lambda_{flow}$ on 306 alone first (same convention as the existing `threshold_strand_coherence`
   sweep), then run all 9 if it looks promising.

---

## Sequencing

```
Day 1       B-0 + B-1 code                              A-0 (2-3 subjects, parallel)  +  A-1 code+sweep
Day 2       B-2a sanity (306, 302) -> gate
Day 2-3     B-2b full 9-subject -> gate
Day 3-5     A-2 (only if A-0 passed + time remains): offline flow cache -> loss wiring -> 306 tuning -> 9-subject run
```

If B-2b's gate fails outright (still net-negative, no improvement over free-vector dynamic), don't
proceed to B-2c — spend the freed time entirely on A instead, since that result would indicate the
missing ingredient is supervision (A), not parameterization (B).