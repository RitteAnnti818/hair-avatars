# Method Equations (for paper Method section)

Notation matches the implementation exactly (`scene/flame_gaussian_model.py`, `scene/gaussian_model.py`,
`train.py`, `arguments/__init__.py`). Cross-reference `related_work_literature_review.md` before
presenting any of this as a "novel mechanism" — see framing recommendation at the end of that file.

## Notation

| Symbol | Meaning | Code |
|---|---|---|
| $f(i)$ | face (triangle) that Gaussian $i$ is bound to | `binding[i]` |
| $\mu_i$ | Gaussian $i$'s local (triangle-relative) position | `_xyz[i]` |
| $R_{f}(t), k_f(t), T_f(t)$ | face orientation, scale, world centroid at time $t$ | `face_orien_mat`, `face_scaling`, `face_center` |
| $\mathcal{F}_{hair}$ | hair-region faces, 656 / 10144 | `flame.mask.f.hair` |
| $K$ | number of strands | 32 |
| $\mathcal{S}_s$ | ordered face chain of strand $s$ (root$\to$tip) | `phase1_strands_k32.json` |
| $n_s$ | chain length of strand $s$ | `strand_chain_len[s]` |
| $j \in \{0,\dots,n_s-1\}$ | link index within a strand (0 = root/scalp, $n_s{-}1$ = tip) | `face_chain_pos` |
| $\bar\delta_{s,j}$ | static (time-invariant) per-link offset | `_strand_delta_static[s,j]` |
| $\epsilon_{s,j}(t)$ | dynamic (time-varying) per-link offset, ablatable | `_strand_delta_dynamic[s,j,t]` |
| $w_{s,j}$ | root$\to$tip ramp weight | computed in `_update_strand_delta_cumsum` |
| $\Delta_{s,j}(t)$ | accumulated chain offset (local frame) | `strand_delta_cumsum[s,j]` |
| $\tau_{static},\tau_{dynamic},\tau_{coh}$ | thresholds | `threshold_strand_{static,dynamic,coherence}` |
| $\lambda_{static},\lambda_{dynamic},\lambda_{coh}$ | loss weights | `lambda_strand_{static,dynamic,coherence}` |
| $m(t)$ | per-frame head-motion magnitude (pose velocity) | `FlameGaussianModel._compute_motion_gate` |
| $g(m(t))$ | motion gate applied to the dynamic offset | `FlameGaussianModel._compute_motion_gate` |
| $m_{ref}$ | reference motion magnitude for gate calibration | `FlameGaussianModel.motion_gate_ref` |

## 0. Baseline rigid binding (GaussianAvatars, unchanged, for reference)

$$
\mu_i'(t) = k_{f(i)}(t)\, R_{f(i)}(t)\, \mu_i + T_{f(i)}(t)
$$
$$
s_i'(t) = k_{f(i)}(t) \cdot s_i, \qquad q_i'(t) = q_{f(i)}(t) \otimes q_i
$$

## C1 — Strand-Chain Soft-Rigging

**Strand topology** (precomputed once, Phase 1): $\mathcal{F}_{hair}$ is partitioned via k-means (on face
centroids) into $K{=}32$ strands $\mathcal{F}_{hair} = \bigsqcup_{s=1}^{K}\mathcal{S}_s$; faces within each
strand are ordered root$\to$tip by distance from a crown landmark, giving each face a chain index $j$.

**Per-link offset, with motion-gated dynamic term:**
$$
\delta_{s,j}(t) = \bar\delta_{s,j} + g(m(t))\cdot\epsilon_{s,j}(t) \in \mathbb{R}^3
$$

where the **static** term $\bar\delta_{s,j}$ and gate function are:

**Head-motion magnitude** (finite-difference pose velocity — PROPOSED):
$$
m(t) = \left\lVert \begin{bmatrix}\text{rotation}(t)\\ \text{neck\_pose}(t)\end{bmatrix}
- \begin{bmatrix}\text{rotation}(t-1)\\ \text{neck\_pose}(t-1)\end{bmatrix} \right\rVert_2
$$

**Motion gate** (PROPOSED — normalized, clamped to $[0,1]$; $m_{ref}$ calibrated per-subject, e.g. a
high percentile of $m(t)$ over the training sequence):
$$
g(m(t)) = \operatorname{clip}\!\left(\frac{m(t)}{m_{ref}},\ 0,\ 1\right)
$$

At $m(t) \to 0$ (near-static frames, e.g. most of EMO/EXP) the gate closes and the offset reduces to the
static-only case ($\delta_{s,j}(t) \to \bar\delta_{s,j}$) — safe by construction. At $m(t) \geq m_{ref}$
(fast head motion) the gate opens fully and the raw (previously-shown-harmful) dynamic term is allowed
to act at full strength. $m_{ref}$ = per-subject 90th percentile of $m(t)$ over that subject's own
training sequence (fixed rule, same percentile for every subject — not tuned per-subject to maximize
held-out metrics).

**Result (tested on all 9 subjects, FREE sequence, threshold_strand_coherence=0.2): the hypothesis above
was falsified beyond the single case that inspired it.** Net effect across 9 subjects is NEGATIVE
(precise-hair-crop $\Delta$PSNR: simple avg $-0.080$dB, frame-weighted avg $-0.077$dB; full-image avg
$-0.074$dB), improving only 264 ($+0.195$dB) and 218 ($+0.141$dB), with a severe regression on 304
($-0.525$dB, only 15.8% better-frames). Motion-profile peakiness (p99/median of $m(t)$) does **not**
predict outcome: 304 ranks #2/9 in peakiness (behind only 264) yet is the worst result, while 218 ranks
#7/9 in peakiness yet is the second-best result. static-only remains the better default (positive on 8/9
subjects — see `final_results_summary.json`'s `precise_hair_crop_metrics_vs_baseline`). 304's failure is
not yet explained; leading hypothesis is FLAME-tracking noise (304 has the lowest full-image baseline
PSNR of all 9 subjects, 19.958dB, suggesting lower capture/tracking quality, which would inject noise
directly into $m(t)$ since it's a raw frame-to-frame pose delta) — not yet verified. Full numbers:
`final_results_summary.json`'s `motion_gate_full_image_metrics` / `motion_gate_precise_hair_crop_metrics_vs_baseline`
/ `motion_gate_verdict`.

**Ablation status of the dynamic term:**

| Variant | Status | 306 precise-crop $\Delta$PSNR | 9-subject avg precise-crop $\Delta$PSNR |
|---|---|---|---|
| Naive dynamic ($g \equiv 1$, unconditional) | tested | $-0.004$dB (harmful) | — |
| Static-only ($\epsilon \equiv 0$, i.e. $g \equiv 0$) | tested | $+0.281$dB | positive on 8/9 subjects |
| **Motion-gated dynamic (above)** | **tested, net negative** | $-0.089$dB | $-0.080$dB (2/9 subjects improved) |

**Dev-sweep: strand count $K$ and magnitude threshold $\tau_{static}$ (static-only, no dynamic term).**
Tested whether the adopted $K{=}32$, $\tau_{static}{=}0.3$/$\lambda_{static}{=}10^{-2}$ config (already
chosen via the coherence-threshold sweep) is actually optimal, by trying $K{=}64$ (finer clustering,
same k-means+crown-distance method) and a relaxed magnitude cap ($\tau_{static}{=}0.6$,
$\lambda_{static}{=}3\times10^{-3}$) on 306/302 (already-positive under the adopted config) and 074
(the known hairstyle-mismatch failure, see `open_anomaly_074`).

| Subject | Adopted (K=32, $\tau{=}0.3$) | K=64 | Relaxed $\tau_{static}$ |
|---|---|---|---|
| 306 | $+0.281$dB | $-0.093$dB (worse) | $+0.178$dB (worse) |
| 302 | $+0.014$dB | $-0.233$dB (worse) | $-0.208$dB (worse) |
| 074 | $-0.138$dB (failure) | $-0.041$dB (less bad) | $+0.014$dB (**sign flip**) |

Both changes make the already-good subjects (306, 302) worse — the adopted config sits near a local
optimum for them, and moving either axis away from it hurts. But both changes *help* 074, and the
relaxed threshold flips 074's known failure to non-negative. **No single global $(K,\tau_{static})$
helps every subject** — this is empirical evidence for the already-flagged future-work direction
(adaptive/subject-specific strand topology or per-subject magnitude threshold), not a case for changing
the global default. Full numbers: `final_results_summary.json`'s `csc_devsweep_full_image_metrics` /
`csc_devsweep_precise_hair_crop_metrics_vs_baseline` / `csc_devsweep_verdict`.

**Dev-sweep: chain-propagated rotation (third independent capacity expansion, same pattern).**
Extended C1 with a per-link rotation chain — axis-angle $\mathrm{rot}_{s,j}\in\mathbb{R}^3$, composed via
quaternion product with the same root-fixed ramp as position ($Q_j = Q_{j-1}\otimes\exp(w_j\,\mathrm{rot}_j)$,
$Q_0=I$) — tested on all 9 subjects against the adopted static-only baseline. Net negative
(avg $-0.14$dB over 8 subjects with a valid comparison), only 3/9 improved (253, 074, 218). Same
signature as the K/threshold sweep: `corr(existing static-only delta, rotation delta) = -0.756`
(n=8, p<0.05) — subjects already well-fit by minimal capacity (306, 104) are hurt most by added DOF;
074 (topology-mismatched) is helped most. **This is the third independent geometric-capacity expansion
(resolution, magnitude threshold, rotation DOF) showing the identical pattern** — strong evidence the
adopted minimal config is near a local optimum for well-matched hairstyles, not under-parameterized.
Post-hoc note: the 3-DOF axis-angle used here is more capacity than Discrete Elastic Rods' material
frame actually prescribes (1-DOF twist about the tangent; the other 2 DOF are determined by the
tangent itself) — a twist-only reimplementation is a plausible fix (less capacity to overfit) but not
yet tested. Full numbers: `final_results_summary.json`'s `strandrot_precise_hair_crop_vs_static_only`.

**Dev-check: is permanent triangle binding itself the bottleneck? (TeGA, arXiv:2505.05672)**
TeGA characterizes GaussianAvatars' binding as "a greedy and potentially suboptimal solution" and lets
Gaussians move continuously across triangles via a UVD-space + per-Gaussian-Jacobian reparameterization
(requires a full UV texture parameterization and a deformation U-Net — a different architecture, not a
flag on top of ours). As a much lighter proxy, tested periodic (every 5000 iters) nearest-hair-triangle
reassignment for hair Gaussians with **no strand chain** (`enable_rebinding`, isolated from C1/C2), on
subject 306: **+0.050dB** vs rigid baseline (51.5% better-frames) — recovers only ~18% of the static-only
chain's gain (+0.281dB). Confirms binding rigidity is *a* contributing factor (TeGA's premise holds
directionally) but shows the chain-offset structure itself is responsible for most of C1's benefit, not
mere unbinding. Useful for the novelty argument: C1 is not a hair-specific rediscovery of TeGA's
rebinding effect. Full numbers: `final_results_summary.json`'s `rebinding_vs_chain_306`.

**Root-fixed, tip-increasing freedom ramp:**
$$
w_{s,j} = \operatorname{clip}\!\left(\frac{j}{n_s - 1},\ 0,\ 1\right) \qquad (w_{s,0}=0,\ w_{s,n_s-1}=1)
$$

**Chain accumulation** (cumulative sum along the strand, root anchored):
$$
\Delta_{s,j}(t) = \sum_{k=0}^{j} w_{s,k}\, \delta_{s,k}(t), \qquad \Delta_{s,0}(t) = 0
$$

**Applied on top of the rigid binding**, for Gaussian $i$ bound to face $f(i) \in \mathcal{S}_s$ at
chain position $j$ (hair-region Gaussians only; non-hair Gaussians are untouched):
$$
\mu_i'(t) = \underbrace{k_{f(i)}(t)\,R_{f(i)}(t)\,\mu_i + T_{f(i)}(t)}_{\text{rigid (unchanged)}}
\;+\; \underbrace{k_{f(i)}(t)\,R_{f(i)}(t)\,\Delta_{s,j}(t)}_{\text{C1 hair offset}}
$$

Note the offset is rotated by the *same* face frame $R_{f(i)}(t)$ and scaled by the *same* $k_{f(i)}(t)$
as the rigid term — required so the offset lives in the same "triangle-relative units" convention as
$\mu_i$; omitting the $k_{f(i)}(t)$ factor was an earlier bug (effective freedom then depended on each
triangle's absolute scale, breaking cross-subject comparability).

## C2 — Strand-Coherence Regularization

Penalizes relative displacement between **adjacent** links on the same chain, beyond a threshold:
$$
\mathcal{L}_{coh} = \frac{1}{|\mathcal{N}|} \sum_{s=1}^{K} \sum_{j=1}^{n_s - 1}
\operatorname{ReLU}\!\Big(\big\lVert \Delta_{s,j}(t) - \Delta_{s,j-1}(t) \big\rVert_2 - \tau_{coh}\Big)
$$
where $\mathcal{N} = \sum_{s=1}^K (n_s-1)$ counts only valid (non-padding) adjacent pairs — the dense
tensor is padded to $\max_s n_s$, so pairs beyond a given strand's true length are excluded. Tuned
value: $\tau_{coh} = 0.2$ (swept over $\{0.1, 0.15, 0.2, 0.3, 0.5\}$ on subject 306; see
`final_results_summary.json`'s `threshold_sweep_note`).

This is distinct from the two magnitude-shrinkage terms below: $\mathcal{L}_{coh}$ bounds how much two
*neighboring* links may move relative to each other, independent of either one's absolute offset size.

## Existing per-link magnitude losses (reused GaussianAvatars threshold-ReLU pattern)

$$
\mathcal{L}_{static} = \frac{1}{K \cdot n_{\max}} \sum_{s,j} \operatorname{ReLU}\big(\lVert \bar\delta_{s,j} \rVert_2 - \tau_{static}\big), \qquad \tau_{static}=0.3
$$
$$
\mathcal{L}_{dynamic} = \frac{1}{K \cdot n_{\max} \cdot T} \sum_{s,j,t} \operatorname{ReLU}\big(\lVert \epsilon_{s,j}(t) \rVert_2 - \tau_{dynamic}\big), \qquad \tau_{dynamic}=0.05
$$
($n_{\max} = \max_s n_s$; $T$ = number of timesteps. $\mathcal{L}_{dynamic}$ is inactive when the dynamic
term is disabled for the static-only ablation.)

## Total training objective

$$
\mathcal{L} = (1-\lambda_{ssim})\,\mathcal{L}_1 + \lambda_{ssim}\,(1 - \mathrm{SSIM})
+ \lambda_{xyz}\mathcal{L}_{xyz} + \lambda_{scale}\mathcal{L}_{scale}
+ \lambda_{static}\mathcal{L}_{static} + \lambda_{dynamic}\mathcal{L}_{dynamic} + \lambda_{coh}\mathcal{L}_{coh}
$$

$\mathcal{L}_1$, SSIM, $\mathcal{L}_{xyz}$, $\mathcal{L}_{scale}$ are the unmodified GaussianAvatars terms
($\lambda_{ssim}=0.2$). Hyperparameters for the new terms:

| Term | $\lambda$ | $\tau$ |
|---|---|---|
| $\mathcal{L}_{static}$ | $10^{-2}$ | $0.3$ |
| $\mathcal{L}_{dynamic}$ | $10^{-1}$ | $0.05$ |
| $\mathcal{L}_{coh}$ | $10^{-2}$ | $0.2$ (tuned; default was $0.1$) |

Optimizer: strand parameters $\{\bar\delta, \epsilon\}$ get their own Adam param group,
`strand_delta_lr` $=10^{-3}$, separate from the base Gaussian xyz/scale/rotation learning rates.

## Framing reminder for the paper

Per the literature review, neither the root-fixed ramped-chain structure ($w_{s,j}$, $\Delta_{s,j}$) nor
the adjacent-link threshold penalty ($\mathcal{L}_{coh}$) is a novel *mechanism* in isolation — both are
standard rod/hair-simulation constructs (Bergou et al. 2008 DER, Bertails et al. 2006 Super-Helices,
Selle et al. 2008 mass-spring hair, Provot 1995 strain limiting). Present C1/C2 as adapting this known
mechanism lineage into a lightweight, simulation-free, *learned* offset chain on top of GaussianAvatars'
rigid binding — the novelty claim is the instantiation/combination, not the underlying math.
