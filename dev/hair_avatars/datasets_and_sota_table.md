# Datasets and Main-Table SOTA (2026-08-01)

Follow-up to `related_work_literature_review.md` (2026-07-26/28), which covers competing
mechanisms/novelty. This doc covers: (1) datasets used across this literature, (2) which methods
belong in the main quantitative comparison tables. Sourcing standard: quotes fetched directly from
arXiv/CVF HTML where possible; anything only found via search snippet is marked **미확인**.

---

## 1. Datasets used across this literature

### Multi-view, rig-based (our family — GaussianAvatars lineage)

| Dataset | Used by | 공개 여부 (Public?) | Spec (quoted where available) | Notes |
|---|---|---|---|---|
| **NeRSemble** | GaussianAvatars (CVPR'24, our baseline), HHAvatar (CVPR'24), FHAvatar (CVPR'26), GHA (CVPR'24), MeGA (CVPR'25), FlashAvatar (CVPR'24), NPGA (SIGGRAPH Asia'24), SurFhead (ICLR'25), GEM (CVPR'25), RGBAvatar (arXiv'25), TexAvatars (3DV'26) | **공개 (gated)** — request form (`forms.gle/rYRoGNh2ed51TDWX9`) + ToS agreement, approval "typically within 1 day," download via [tobias-kirschstein/nersemble-data](https://github.com/tobias-kirschstein/nersemble-data). | GaussianAvatars: *"video recordings of 9 subjects... All recordings contain 16 views... 11 video sequences... 802×550"*. TexAvatars: *"training uses 10 corpora and 15 cameras, holding out 1 near-frontal camera for novel-view testing, while a single corpus is reserved for novel-expression evaluation"* — states explicitly it follows GaussianAvatars' protocol. | The de facto standard dataset for this entire comparison space; several different subsetting conventions exist across papers — state protocol explicitly when citing. |
| **Ava-256** (Meta, Codec Avatar Studio, Martinez et al., NeurIPS'24 D&B) | PhysHead (CVPR'26) | **완전 공개** — [facebookresearch/ava-256](https://github.com/facebookresearch/ava-256), public AWS S3, `download.py`, CC-BY-NC 4.0, 256 subjects, no gated form. | *"We use the Ava-256 dataset with 16 cameras per subject, selected via the Hungarian algorithm to match the Nersemble dataset distribution"* | Most frictionless access at scale vs. NeRSemble's gated ~9-10 subjects — worth considering as a secondary dataset. |
| **Self-captured (HHAvatar)** | HHAvatar (CVPR'24) | **비공개** | *"there is no complete non-rigid dynamic hair trajectory (e.g., nodding and swinging at different speeds) in NeRSemble"* → shot their own 4-camera 4K set to get it | Direct primary-source confirmation that **NeRSemble lacks adequate hair-dynamics content** — citable caveat for us too. |
| **Goliath Dataset** (Meta) | RGCA (CVPR'24) | **비공개 (gated)** — email request to a named Meta researcher, institutional email preferred | — | Different ecosystem from NeRSemble/Ava-256; RGCA is not directly comparable to our NeRSemble-based work for this reason. |

### Hair-specific reconstruction

| Dataset | Used by | 공개 여부 (Public?) | Notes |
|---|---|---|---|
| Synthetic (Blender-simulated, self-made) | DGH (NeurIPS'25) | **비공개/미확인** | *"we simulate 100 motion sequences of 100 frames... 1500k hair strand with 24 vertices per strand"*, 5 hairstyle subjects. **No real captured video anywhere in DGH** — purely synthetic. |
| Own captures (MPI-IS / Samsung AI Center) | Gaussian Haircut (ECCV'24) | **비공개** | No public dataset name/link found. |
| **RealHair** (self-introduced) | GaussianHair (arXiv'24, venue unconfirmed) | **불확실** — repo public, code "Coming soon" | 281 mobile-phone videos. Static reconstruction; "dynamics" = post-hoc CG-engine animation, not learned from real motion. |
| Studio hair/hairless paired captures | HairCUP (ICCV'25) | **비공개로 추정** | Meta/Reality Labs authors; reads as proprietary studio data, not confirmed released. |
| Unconfirmed | HADES (ICCV'25) | **미확인** | Check before citing numbers. |
| **Strands400** (self-introduced) | GeomHair (arXiv'25, May) | **공개로 보임** | 400 subjects' hair-strand geometry, GDPR-compliant release process. Built from NPHM's 383 scans + 17 newly collected. **Colorless 3D scans only — no multi-view RGB video, no motion/dynamics.** Doesn't compete with NeRSemble for our use case (needs dynamic hair motion), but is a genuine new public entry for static hair geometry. |

**2026-08-05 재조사 (WebSearch, CVF page returned 403 so HADES's own dataset section still unconfirmed):**
- **HairGS** (arXiv'25, Sep) confirms it's tested on NeRSemble — the pattern holds into 2025 papers too, no sign of NeRSemble losing dominance.
- **HADES** (ICCV'25): couldn't fetch primary source directly (403 on CVF page). Secondary search snippets describing "self-captured data, ~1000 frames, 4 cameras at ~90°, 4K" strongly resemble HHAvatar's own self-capture setup — HADES likely follows the same NeRSemble + self-captured-supplement pattern as HHAvatar, but this is **not primary-source-confirmed**, so keep it as "미확인" until verified directly.
- No new fully-public **dynamic multi-view hair-motion** dataset was found to rival/replace NeRSemble. Strands400 (above) is the only new public dataset found, and it's static geometry, not motion capture.
- **Caveat on this re-check itself**: still a targeted search against known paper names, not an exhaustive scan of the entire head-avatar/hair-reconstruction literature — treat "NeRSemble is essentially the only public option for dynamic multi-view hair capture" as well-supported but not proven exhaustively.

**2026-08-05: how many NeRSemble subjects does each paper actually use? (WebFetch on arXiv HTML, primary source)**

| Paper | Subjects used | Source |
|---|---|---|
| GaussianAvatars | **9** | Official repo/paper, confirmed earlier. |
| **NPGA** | **6** (3 for ablations) | Direct quote from arxiv.org/html/2405.19331 §5: *"we choose a diverse set of six subjects performing challenging facial expressions"*; §5.5: *"we perform ablation experiments using three subjects."* No subject IDs listed. |
| **TexAvatars** | **미확인 (not stated as a number)** | Direct quote from arxiv.org/html/2512.21099v1: *"Expression corpora are drawn from ten unique motion sequences, each capturing variations in facial actuation."* — confirms "10 corpora" = **10 sequences per subject**, NOT 10 subjects (an earlier WebSearch snippet misread this as "10 subjects" — wrong, corrected here). Paper says it follows GaussianAvatars' exact protocol but never states its own subject count or IDs anywhere I could fetch. |
| **GEM** | 미확인 | A WebSearch snippet claimed "same 9 subjects as GaussianAvatars" but this was not traced to primary text — treat as unverified. |

Total dataset size (NeRSemble v2) also re-verified directly against our own `nersemble-data` tool (authoritative, not the launch-announcement tweet): `nersemble-data download ... --participant all` → **"Selected 418 participants"** (2026-08-05). The v2 launch tweet said 425; the discrepancy is most likely post-launch participant consent withdrawals (GDPR right-to-deletion — TUM is an EU institution, and we saw the same clause explicitly in the Strands400 release). **418 is the current real number**, not 425.

Bottom line: across every paper we could verify, subject counts sit in the **6–9 range**, far below NeRSemble v2's 418 available participants — reinforcing the point above (per-subject compute cost dominates the field's practice, not data availability).

**2026-08-05: is a single sequence ever split between train and eval (partial use) in this literature?**

Checked our own actual `UNION10_074_...` baseline data plus NPGA/TexAvatars/GEM primary/secondary text. **No — every case found holds out an entire whole sequence, never a temporal (frame-level) split within one sequence:**

| Source | Evidence |
|---|---|
| Our own `UNION10_074_...` data | `sequences_test.txt` = `EMO-4` (whole sequence); `sequences_trainval.txt` = the other 9 sequences (whole). Confirmed by inspecting the actual files on disk, not a claim from a paper. |
| **NPGA** | Primary quote (arxiv.org/html/2405.19331): *"all avatars are trained on a set of 21 training sequences... evaluated using... a held-out test sequence"*; *"we train our avatars on all sequences, except for the 'FREE'-sequence which we keep as a held-out evaluation sequence for the self-reenactment task."* |
| **TexAvatars** | Primary quote (arxiv.org/html/2512.21099v1): *"one sequence for each subject is randomly held out as a self-reenactment evaluation set."* |
| **GEM** | Search-engine paraphrase only (not independently fetched/quoted from GEM's own PDF) — wording suspiciously mirrors NPGA's, so treat as **weak/unverified**, likely just describing the shared field convention rather than GEM's own confirmed text. |

**Implication for our hair-dynamics contribution**: this whole-sequence-holdout convention doesn't give a clean way to both (a) let HAIR contribute training signal for dynamics learning and (b) get a dedicated quantitative held-out metric for hair-dynamics reconstruction — the two established options (hold HAIR out entirely as the test sequence vs. put it entirely in train) each sacrifice one side. Splitting HAIR itself temporally (train on early frames, held-out eval on later frames of the *same* sequence) is not something we found precedent for in this literature — it would be a deliberate departure from convention, justified specifically because our contribution's axis of novelty (hair dynamics over time) doesn't map onto the field's existing axis (held-out whole *expression type*).

### FINAL DECISION (2026-08-05): our train/eval split

Resolved without departing from the field's whole-sequence-holdout convention, by exploiting a property of
our own data curation choice: our 5 target subjects are all drawn from the "hair-down women" subset (see
[[nersemble-woman104-target]]), so **HEAD (EXP-1-head) also contains genuine hair swing motion** from
natural head-turning, not just HAIR — it's not just head-pose variation like it is for a typical (hair-tied
or short-hair) NeRSemble subject.

| Role | Sequences |
|---|---|
| **Train** | UNION10 (all 10: EMO-1,2,3,4 + EXP-2,3,4,5,8,9) + HAIR (full) |
| **Hair-dynamics eval (our contribution's metric)** | HEAD, held out entirely |
| **Novel-expression eval (baseline-comparable)** | FREE, held out entirely |
| **NVS eval** | 1 camera auto-held-out across the train pool (`combine_nerf_datasets.py` default) |

Why this works: HEAD was never part of GaussianAvatars' own UNION10 protocol to begin with, so holding it
out costs us nothing from the standard-comparable UNION10 training data (all 10 stay in train, unlike
baseline reproductions that sacrifice one, e.g. EMO-4 for subject 074). HAIR gets to fully contribute
training signal. Evaluating hair reconstruction on HEAD — a sequence with real hair motion the model never
trained on — is a genuine generalization test of the dynamics model, not just held-out camera/frames of the
same captured motion. Still fully whole-sequence-holdout, so methodologically consistent with
GaussianAvatars/NPGA/TexAvatars precedent, just with subject selection (hair-down women) doing the work
that lets HEAD double as a meaningful hair-dynamics test set.

### Bottom line for our own dataset choice
We already train on **NeRSemble** (`UNION10EMOEXP_*`, GaussianAvatars' 9-subject/16-camera protocol) —
this is the dataset the large majority of relevant comparisons above also use, and it keeps us
directly comparable to them. HHAvatar's own primary-source complaint about NeRSemble's hair-motion
coverage is worth citing as a limitation-section caveat regardless of our results.

---

## 2. Main comparison tables

### Table A — General avatar quality (NeRSemble)

**⚠️ IMPORTANT — read before citing any number below as "SOTA": there is no single comparable
number across this literature.** Verified directly from primary text (2026-08-04): three different
papers report their own GaussianAvatars *reproduction* at wildly different PSNR depending on
evaluation protocol — the same nominal method spans nearly 10dB:

| Source paper | Protocol | GaussianAvatars PSNR (as reproduced by that paper) |
|---|---|---|
| TexAvatars | Novel Expression (FREE split) | 22.01 |
| TexAvatars | Novel Expression (Held-out) | 25.00 |
| NPGA | Self-reenactment (held-out sequence) | 27.77 |
| TexAvatars | Novel View (held-out camera) | 29.23 |
| GEM | Novel Expression + View combined | 31.32 |

**Do not rank methods by bare PSNR across papers — protocol alone moves the number by ~10dB.**
Any Table A in our paper must either (a) report our own reproduced GaussianAvatars baseline as the
sole anchor and compare only within our own protocol, or (b) explicitly label every borrowed number
with its source paper's exact protocol name, never as a flat ranked list.

**Per-paper verified numbers** (primary text, not search snippets):

| Method | Venue | Protocol | PSNR | SSIM | LPIPS | Notes |
|---|---|---|---|---|---|---|
| GaussianAvatars | CVPR'24 | — | — | — | — | Our baseline / anchor row (use our own reproduction, not a borrowed number). |
| **GEM** | CVPR'25 | Novel Expression + View | 32.68 | 0.9633 | 0.0675 | Corrected 2026-08-04 from an unverified snippet (was mis-recorded as 33.55/0.966/0.068). GA reproduced at 31.32 in the same table; Animatable Gaussians 32.41; INSTA 27.78. |
| **NPGA** | SIGGRAPH Asia'24 | Self-reenactment (held-out seq.) | 30.26 | 0.934 | 0.055 | GA reproduced at 27.77 in the same table (self-reenactment specifically — lower than GEM's protocol). Abstract's "≈2.6 PSNR over prior SOTA" = vs. GaussianHeadAvatar (26.81), not vs. GA. |
| **TexAvatars** | 3DV'26 | Novel Expression (Held-out) | 25.61 | 0.894 | 0.048 | GA reproduced at 25.00 in the same protocol/table. |
| **TexAvatars** | 3DV'26 | Novel Expression (FREE split) | 22.84 | 0.861 | 0.077 | GA reproduced at 22.01. FREE-split data already available locally (`data/*_FREE_v16_...`, 14 subjects) — reproducible without new data. |
| **TexAvatars** | 3DV'26 | Novel View (held-out camera) | 35.15 | 0.947 | 0.030 | GA reproduced at 29.23. Highest raw PSNR of any entry here — but easiest protocol (novel view, not novel expression), not comparable to GEM/NPGA's numbers. |
| **MeGA** | CVPR'25 | — | — | — | — | Co-authored by Shenhan Qian (GaussianAvatars' own first author). Hybrid mesh (face) + Gaussian (hair, with deformation field). Numbers not yet verified from primary text. |
| **FlashAvatar** | CVPR'24 | — | — | — | — | Confirmed exact protocol (9 subjects, 16 viewpoints) but numbers not yet verified. |
| **GHA** (Gaussian Head Avatar) | CVPR'24 | — | — | — | — | Concurrent with GaussianAvatars; never directly cross-compared in either paper — a gap we could fill. Numbers not yet verified. |
| **SurFhead** | ICLR'25 | — | — | — | — | Geometry/normal-accuracy focused (2D Gaussian surfels), not hair-specific. Numbers not yet verified. |
| **RGBAvatar** | CVPR'25 | — | — | — | — | Real-time/online reconstruction. Numbers not yet verified. |
