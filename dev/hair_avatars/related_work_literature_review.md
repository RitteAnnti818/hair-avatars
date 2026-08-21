# Related Work / Novelty Check — Literature Review (2026-07-26)

Conducted after directly asking "where did the C1/C2 formulas come from" and admitting they were
designed ad hoc during development, without a prior literature search. This review exists to close
that gap before submission. ~20 searches; equations directly fetched/quoted from primary sources
(arXiv HTML) for Gaussian Haircut, PhysHead, DGH, STAvatar. Two items (FHAvatar, GaussianHair) could
only be characterized from search snippets — flagged below for a manual read before submission.

## Motivation framing (added 2026-07-28): three FLAME/NPHM-common failure modes

Verified via direct fetch of each paper's own Limitations section (not paraphrased from search
snippets). Terminology below matches the papers' own wording exactly — do NOT use "eye vergence"
(clinical term, not what either paper says); the verified term is **eye gaze animation /
eyeball rotation**.

| # | Term (use this exact phrasing) | Exact quote | Source |
|---|---|---|---|
| 1 | **tongue articulation** (tongue geometry) | "could not animate unmodeled parts like the tongue" | GaussianAvatar-Editor, arXiv:2501.09978 |
| | | "tongue... cannot be animated as reliably... due to overfitting" | NPGA, arXiv:2405.19331, Sec. 6 (Limitations) |
| 2 | **eye gaze animation** (NOT "vergence") | "eyeball rotation... cannot be animated as reliably" | NPGA, arXiv:2405.19331, Sec. 6 |
| 3 | **hair dynamics** (hair modeling) | "we lack control over areas that are not modeled by FLAME such as hair and other accessories" | GaussianAvatars (our own baseline), arXiv:2312.02069, Sec. 5 (Limitations) |

NPGA's exact sentence (Sec. 6): "regions like the neck, torso, tongue, and eyeball rotation, which are
not explained by NPHM's expression codes, cannot be animated as reliably or might even lead to
artifacts due to overfitting." NPGA does not mention hair in its limitations section; GaussianAvatars
does not mention tongue/eyeball rotation in its limitations section (Section 5 only cites relighting
and "hair and other accessories"). The two sources are complementary, not overlapping — cite both.

**Why this is a stronger motivation than a FLAME-only framing**: tongue and eye-gaze failures are
confirmed on *both* a linear-blendshape rig (FLAME/GaussianAvatars family, via GaussianAvatar-Editor)
and a neural-expression-space rig (NPHM/NPGA) — i.e. this isn't a FLAME-specific defect fixable by
swapping the parametric prior, it recurs across differently-structured low-dimensional expression
codes. Suggested motivation sentence:

> "Parametric Gaussian avatars, whether rigged to a linear blendshape model (GaussianAvatars) or
> conditioned on a neural expression space (NPGA), share a common failure mode: regions not
> well-explained by the underlying model's expression code remain unreliable or unmodeled altogether
> — tongue articulation and eye gaze animation are explicitly acknowledged as such by NPGA, while hair
> dynamics is explicitly acknowledged as unmodeled by GaussianAvatars itself. In this work we focus on
> the third — dynamic hair motion — which persists as an open problem regardless of the choice of
> underlying parametric prior."

Scope reminder: only **hair dynamics** is addressed by this paper's contributions (C1/C2). Tongue and
eye gaze are cited purely as motivating context for "expression-code blind spots are a real, persistent
category of problem" — do not imply either is solved here.

## Our mechanisms, for reference

- **C1 (Strand-Chain Soft-Rigging)**: hair-region FLAME faces (656/10144) clustered into K=32 strands
  (k-means on face centroids, root/tip ordered by distance from crown). Per-link learnable offset
  δ_j(t) = static_j + dynamic_j(t), accumulated along the chain: Δ_j = Δ_{j-1} + w_j·δ_j, Δ_0 = 0
  (root rigid), ramp weight w_j = j/(n_links-1) (0 at root → 1 at tip). Rotated by triangle orientation,
  scaled by triangle scale k_i, added on top of GaussianAvatars' rigid binding.
- **C2 (Strand-Coherence Regularization)**: relu(‖Δ_j − Δ_{j-1}‖ − threshold), averaged over valid
  adjacent link pairs, threshold_strand_coherence=0.2 (tuned from default 0.1).

## Overall verdict

No paper found implements the *combined* C1+C2 system (strand-clustered chain of learnable
static+dynamic offsets, root-fixed, linear ramp, on top of GaussianAvatars' rigid binding, plus a
relu-thresholded adjacent-link coherence loss) as one system. As a specific engineering instantiation
on top of GaussianAvatars, this appears to be a genuinely novel combination.

**But**: (1) the problem statement ("dynamic Gaussian hair on an avatar") is directly contested by a
NeurIPS 2025 paper (DGH) that a reviewer will likely know, and (2) the underlying mechanisms
(root-fixed ramped chain; adjacent-point stretch/coherence penalty) are decades-old, standard
rod/hair-simulation constructs — claiming them as novel *mechanisms* (rather than a novel
*application* of known mechanisms to this setting) would be an overclaim a CG-literate reviewer would
flag. See "Framing recommendation" at the end.

---

## Part 1 — Directly competing problem space (dynamic/rigged Gaussian hair on avatars)

### DGH: Dynamic Gaussian Hair — **highest risk item**
Wang, Xu, Tretschk, Wang, Ianina, Bozic, Neumann, Tung (USC + Meta Reality Labs). NeurIPS 2025,
arXiv:2512.17094 (Dec 2025).
- Strands = "sequence of connected cylindrical Gaussians." Dynamics via a data-driven
  coarse-to-fine MLP (𝒟) trained on synthetic simulation ground truth, predicting per-point
  displacement/flow fields.
- Losses: Stage I `L_total = λp·L_point + λSDF·L_SDF` (point MSE + body-collision SDF, Eq.1);
  Stage II `L = L_flow = MSE(F_flow^t, F_GT^t)` (Eq.2).
- No chain/cumulative-sum structure, no root-to-tip ramp weight, no adjacent-point coherence loss.
- **Same problem setting, different mechanism.** MUST cite and explicitly differentiate: our chain is
  explicit and simulation-free/lightweight vs. their learned, simulation-supervised dynamics field.

### PhysHead: Simulation-Ready Gaussian Head Avatars
Kabadayi, Sklyarova, Zielonka, Thies, Pons-Moll. CVPR 2026, arXiv:2604.06467. (Thies also co-authored
Gaussian Haircut.)
- Hair dynamics driven by an actual physics simulator (Bullet Physics, semi-implicit Euler:
  `v(t+Δt)=v(t)+F(t)/m·Δt`, `x(t+Δt)=x(t)+v(t+Δt)·Δt`, Eqs.1-2). Sparse simulated guide strands
  propagate to dense hair via kNN strand-skinning transferring relative displacement.
  Only strand regularization found is *color*-consistency between neighbors (Eq.6:
  `L_consistency = Σ_i Σ_{j∈N(i)} ‖c_i − c_j‖²`) — appearance, not geometry.
- Same general goal, different mechanism (physics engine vs. learned offset chain; no geometric
  coherence loss). Cite as directly relevant related work.

### 3DGStrands
Dominguez-Elvira, Alfonso-Arsuaga, Barrueco-Garcia, Comino-Trinidad. Computers & Graphics 131 (2025).
- Material Point Method (MPM) physics simulation directly on a 3DGS representation ("first method to
  simulate hair dynamics within a 3DGS model"). Physics-sim based, not chain-offset learning.
- Different mechanism, adjacent problem. Cite for completeness.

### STAvatar: Soft Binding and Temporal Density Control for Monocular 3D Head Avatars
CVPR 2026, arXiv:2511.19854. Directly extends GaussianAvatars' rigid binding.
- Each texel gets one 13-dim offset (δμ, δs, δr, δα, δc) from a UV-space feature map — no chain, no
  strand structure, no root-to-tip ramp; applied uniformly to all mesh regions (wrinkles, teeth, hair
  alike), not hair-specific. Regularization is magnitude-only
  (`L_offset = λ3|δs−1| + λ4·δc`) — no adjacent-Gaussian coherence term.
- Same base framework (GaussianAvatars soft-binding lineage), fundamentally different mechanism
  (generic unstructured per-Gaussian offset vs. strand-organized chain). Good evidence the
  strand-chain-with-ramp idea hasn't appeared even in other GaussianAvatars soft-binding follow-ups.
  Cite as the nearest "soft binding on top of GaussianAvatars" precedent.

### HHAvatar: Gaussian Head Avatar with Dynamic Hairs
Liao et al., arXiv:2312.03029 (2023/2024). Earliest "dynamic hair for a Gaussian head avatar" found.
MLP deformation field + occlusion perception module; not FLAME/GaussianAvatars-rig based, fully
implicit, no strand structure. Cite to motivate "rigid Gaussian hair is a known limitation" framing
in the intro — not a mechanism competitor.

### Need the researcher's own read (agent could not fetch full equations)
- **FHAvatar** (CVPR 2026, arXiv:2603.23345) — strand/hair-aware Gaussian avatar; "dynamics" claim per
  snippets appears to be fast few-shot reconstruction + real-time FLAME-driven replay, not confirmed.
- **HairCUP** (ICCV 2025, arXiv:2507.19481) — compositional face/hair *appearance* prior, likely not a
  chain-rigging competitor, but check.
- **GaussianHair** (Luo et al., arXiv:2402.10483) — "each strand as a sequence of connected cylindrical
  3D Gaussian primitives," blurb explicitly claims "dynamic animation capabilities." Plausible near-miss
  — **read this one first**.

---

## Part 2 — Gaussian Haircut (explicitly checked, since already on local disk)

Zakharov, Sklyarova, Black, Nam, Thies, Hilliges. ECCV 2024, arXiv:2409.14778.
"Human Hair Reconstruction with Strand-Aligned 3D Gaussians."

Quoted from paper (Sec 3.2): strands are polylines `S_k = {p_l^k}`; Gaussians attached per
line-segment `{p_l^k, p_{l+1}^k}`, scale `s_l^k = {½·‖p_{l+1}^k − p_l^k‖₂, ε, ε}`. Static geometric
attachment rule (segment length → Gaussian scale), **no chain/cumulative-sum offset, no root-to-tip
ramp**. Loss (Eq.8) = photometric + diffusion-prior/SDS terms — **no adjacent-point coherence/stretch
loss**.

**Verdict: not actually related beyond both mentioning hair/strands.** Different problem too (static
monocular multi-view reconstruction, no rig, no animation). Cite as the most prominent
"Gaussians + hair strands" paper / likely lineage ancestor, but poses no novelty risk to C1/C2.

---

## Part 3 — Classic hair/rod simulation literature (the generic mechanisms, Gaussian-independent)

Neither mechanism is novel in isolation; both are textbook rod/cloth-simulation constructs with
20-30 years of prior art. Our actual novelty (if any) is the specific lightweight instantiation as a
*learned* offset chain bolted onto GaussianAvatars' rigid binding — not the chain-with-ramp or
stretch-penalty concepts themselves.

**(a) Root-fixed chain with root→tip freedom ramp** (Δ_j = Δ_{j-1} + w_j·δ_j, Δ_0=0, w_j: 0→1):
- Bergou, Wardetzky, Robinson, Audoly, Grinspun. "Discrete Elastic Rods." SIGGRAPH/TOG 2008. —
  clamped (root-fixed) node-chain centerline, the canonical rod-chain formulation.
- Bertails, Audoly, Cani, Querleux, Leroy, Lévêque. "Super-Helices for Predicting the Dynamics of
  Natural Hair." SIGGRAPH/TOG 2006. — chain of piecewise-helical segments, clamped at scalp. Closest
  classic analog to a root-fixed chain of per-link DOFs.
- Selle, Lentine, Fedkiw. "A Mass Spring Model for Hair Simulation." SIGGRAPH/TOG 2008. — root-anchored
  mass-spring chains, root-to-tip stiffness taper is standard practice in this lineage.
- Müller, Kim, Chentanez. "Fast Simulation of Inextensible Hair and Fur." VRIPHYS 2012. — PBD-based
  inextensible strand chains, root anchored to scalp.
- Δ_j = Δ_{j-1} + w_j·δ_j is also mathematically identical to forward-kinematic chain composition:
  Magnenat-Thalmann, Laperrière, Thalmann. "Joint-Dependent Local Deformation for Hand Animation and
  Object Grasping." Graphics Interface 1988 (origin of linear blend skinning). Lewis, Cordier, Fong.
  "Pose Space Deformation." SIGGRAPH 2000.
- Non-academic reference point: Maya `nHair` has shipped a root-to-tip "Stiffness Scale" ramp attribute
  since the 2000s — same idea as w_j, in production VFX tooling.

**(b) Penalty on ‖Δ_j − Δ_{j-1}‖ between adjacent chain points** (inextensibility/stretch/coherence):
- Bergou et al. 2008 (DER) — explicit inextensibility constraint `C(x_i,x_{i+1}) = ‖x_{i+1}-x_i‖ - d_i`.
  Essentially the unthresholded, hard-constraint version of our relu-based soft penalty.
- Müller, Heidelberger, Hennix, Ratcliff. "Position Based Dynamics." J. Visual Communication and Image
  Representation 18(2), 2007. General PBD stretch constraint `C_stretch = |p1-p2| - d`.
- Provot. "Deformation Constraints in a Mass-Spring Model to Describe Rigid Cloth Behavior." Graphics
  Interface 1995. **Closest classic match to our specific thresholded form**: strain-limiting
  correction only active once a slack/threshold is exceeded — same "no penalty below threshold,
  penalize only the excess" pattern as `relu(‖Δ_j-Δ_{j-1}‖-threshold)`.
- Müller, Kim, Chentanez 2012 (above) also directly targets hair-strand inextensibility this way.

---

## Part 4 — Recent/unindexed 2025-2026 preprint check

Beyond DGH (Dec 2025) and PhysHead (Apr 2026), checked arXiv July 2026 CV listings directly; no
additional dynamic-hair-Gaussian-avatar paper surfaced. Lower-relevance items also found:
ControlHair (arXiv:2509.21541, physics-sim + video diffusion for *rendering* control, not a Gaussian
avatar rig), HairWeaver (arXiv:2602.11117, video-diffusion hair motion synthesis, not Gaussian),
CGHair (arXiv:2604.03716, static hair-card compression, not dynamics). None use the chain/ramp/
coherence mechanism. Recommend re-running an arXiv new-submissions check close to the 3DV deadline
given the pace of this subfield.

---

## Citation checklist for the related-work section

- GaussianAvatars (base method).
- Gaussian Haircut, GaussianHair — Gaussians-on-strands lineage (cite even though not mechanism
  competitors).
- DGH, PhysHead, 3DGStrands — dynamic Gaussian hair, **competing problem statement**, need explicit
  differentiation, not just a citation.
- STAvatar — nearest "soft binding on GaussianAvatars" precedent; shows our strand-structured approach
  differs from the generic-offset alternative already tried in that lineage.
- HHAvatar — motivates "rigid hair is a known limitation" framing.
- Bergou et al. 2008 (DER), Bertails et al. 2006 (Super-Helices), Selle et al. 2008 (mass-spring hair),
  Müller et al. 2007/2012 (PBD, inextensible hair), Provot 1995 (strain limiting) — mechanism-lineage
  citations for C1's chain/ramp and C2's coherence penalty.

## Framing recommendation

Frame C1/C2 as a novel **application/combination** of known chain-rigging and strain-limiting
mechanisms — adapted into a lightweight, simulation-free, *learned* offset chain bolted onto
GaussianAvatars' rigid triangle binding — rather than claiming the mechanisms themselves are new.
Explicitly cite the classic rod/hair-sim lineage (Part 3) as "the mechanism lineage we adapt." This
converts a potential overclaim risk into a strength (shows awareness, correctly scopes the claim) and
should preempt the most likely reviewer pushback alongside explicit differentiation from DGH/PhysHead.

## Still open before submission
- Personally read GaussianHair (arXiv:2402.10483) and FHAvatar (arXiv:2603.23345) full text — agent
  could only characterize these from search snippets, not primary equations.
- Re-check arXiv new submissions close to the 3DV deadline (fast-moving subfield).
