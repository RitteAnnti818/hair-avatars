"""
Generalized version of precise_hair_crop_eval.py: runs the FLAME-hair-vertex-projection crop
comparison for multiple subjects, given their baseline/hair-strand test render directories.
"""
import json, os
import numpy as np
import torch
from PIL import Image
from flame_model.flame import FlameHead

PROJECT = "/home/hee/hee_data/GaussianAvatars"

SUBJECTS = {
    "306": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/fair60k_hairstrand_306/test/ours_60000",
    ),
    "104": dict(
        data_dir=f"{PROJECT}/data/UNION10_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/UNION10EMOEXP_104_eval_600k/test/ours_60000",
        hair_dir=f"{PROJECT}/output/fair60k_hairstrand_104/test/ours_60000",
    ),
    "253": dict(
        data_dir=f"{PROJECT}/data/UNION10_253_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_253/test/ours_60000",
        hair_dir=f"{PROJECT}/output/fair60k_hairstrand_253/test/ours_60000",
    ),
    "306_static": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly60k_306/test/ours_60000",
    ),
    "104_static": dict(
        data_dir=f"{PROJECT}/data/UNION10_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/UNION10EMOEXP_104_eval_600k/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly60k_104/test/ours_60000",
    ),
    "253_static": dict(
        data_dir=f"{PROJECT}/data/UNION10_253_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_253/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly60k_253/test/ours_60000",
    ),
    "306_FREE": dict(
        data_dir=f"{PROJECT}/data/306_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_hairstrand_306/test/ours_60000",
    ),
    # coherence-on (C2 real implementation) vs same baselines as the coherence-off static entries above
    "306_coh": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly_coh60k_306/test/ours_60000",
    ),
    "104_coh": dict(
        data_dir=f"{PROJECT}/data/UNION10_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/UNION10EMOEXP_104_eval_600k/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly_coh60k_104/test/ours_60000",
    ),
    "253_coh": dict(
        data_dir=f"{PROJECT}/data/UNION10_253_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_253/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly_coh60k_253/test/ours_60000",
    ),
    # new subjects: fair60k baseline + static-only (coherence-on by default, current code)
    "264_static": dict(
        data_dir=f"{PROJECT}/data/UNION10_264_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_264/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly60k_264/test/ours_60000",
    ),
    "302_static": dict(
        data_dir=f"{PROJECT}/data/UNION10_302_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_302/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly60k_302/test/ours_60000",
    ),
    "304_static": dict(
        data_dir=f"{PROJECT}/data/UNION10_304_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_304/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly60k_304/test/ours_60000",
    ),
    "460_static": dict(
        data_dir=f"{PROJECT}/data/UNION10_460_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_460/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly60k_460/test/ours_60000",
    ),
    "306_FREE_static": dict(
        data_dir=f"{PROJECT}/data/306_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly60k_306_FREE/test/ours_60000",
    ),
    # B-2a sanity: hard-clamp inextensible chain, static-only (--disable_strand_dynamic), vs. the
    # same free60k rigid baseline used by 306_FREE_static/302 above -- compare delta directly against
    # the free-vector static-only deltas (306: +0.281dB, 302: +0.014dB) already on file.
    "306_inext": dict(
        data_dir=f"{PROJECT}/data/306_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/inext60k_306_FREE/test/ours_60000",
    ),
    "302_inext": dict(
        data_dir=f"{PROJECT}/data/302_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_302/test/ours_60000",
        hair_dir=f"{PROJECT}/output/inext60k_302_FREE/test/ours_60000",
    ),
    # B-2a retry: cap relaxed 0.3 -> 0.5 (see run_inext_b2a_relaxed.sh), testing whether the
    # original cap was simply too tight vs. what the soft-penalty free-vector model actually used.
    "306_inext_relaxed": dict(
        data_dir=f"{PROJECT}/data/306_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/inext60k_relaxed_306_FREE/test/ours_60000",
    ),
    "302_inext_relaxed": dict(
        data_dir=f"{PROJECT}/data/302_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_302/test/ours_60000",
        hair_dir=f"{PROJECT}/output/inext60k_relaxed_302_FREE/test/ours_60000",
    ),
    # threshold_strand_coherence sweep on 306 (default 0.1 regressed badly; testing looser values)
    "306_th015": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_015_306/test/ours_60000",
    ),
    "306_th02": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_02_306/test/ours_60000",
    ),
    "306_th03": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_03_306/test/ours_60000",
    ),
    "306_th05": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_05_306/test/ours_60000",
    ),
    # threshold=0.2 validation on 104/253/460
    "104_th02": dict(
        data_dir=f"{PROJECT}/data/UNION10_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/UNION10EMOEXP_104_eval_600k/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_02_104/test/ours_60000",
    ),
    "253_th02": dict(
        data_dir=f"{PROJECT}/data/UNION10_253_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_253/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_02_253/test/ours_60000",
    ),
    "460_th02": dict(
        data_dir=f"{PROJECT}/data/UNION10_460_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_460/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_02_460/test/ours_60000",
    ),
    # 074 static-only (threshold=0.2)
    "074_static": dict(
        data_dir=f"{PROJECT}/data/UNION10_074_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/UNION10EMOEXP_074_eval_600k/test/ours_60000",
        hair_dir=f"{PROJECT}/output/staticonly60k_074/test/ours_60000",
    ),
    # threshold=0.2 validation on the remaining subjects (264/302/304/306_FREE)
    "264_th02": dict(
        data_dir=f"{PROJECT}/data/UNION10_264_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_264/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_02_264/test/ours_60000",
    ),
    "302_th02": dict(
        data_dir=f"{PROJECT}/data/UNION10_302_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_302/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_02_302/test/ours_60000",
    ),
    "304_th02": dict(
        data_dir=f"{PROJECT}/data/UNION10_304_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_304/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_02_304/test/ours_60000",
    ),
    "306_FREE_th02": dict(
        data_dir=f"{PROJECT}/data/306_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/cohthresh_02_306_FREE/test/ours_60000",
    ),
    # motion-gated dynamic term (g(m(t)) = clip(m(t)/m_ref_p90, 0, 1)), FREE sequence, vs baseline
    "264_motiongate": dict(
        data_dir=f"{PROJECT}/data/264_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_264/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_motiongate_264/test/ours_60000",
    ),
    "306_motiongate": dict(
        data_dir=f"{PROJECT}/data/306_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_motiongate_306/test/ours_60000",
    ),
    "304_motiongate": dict(
        data_dir=f"{PROJECT}/data/304_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_304/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_motiongate_304/test/ours_60000",
    ),
    "460_motiongate": dict(
        data_dir=f"{PROJECT}/data/460_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_460/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_motiongate_460/test/ours_60000",
    ),
    "074_motiongate": dict(
        data_dir=f"{PROJECT}/data/074_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_074/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_motiongate_074/test/ours_60000",
    ),
    "104_motiongate": dict(
        data_dir=f"{PROJECT}/data/104_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_104/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_motiongate_104/test/ours_60000",
    ),
    "218_motiongate": dict(
        data_dir=f"{PROJECT}/data/218_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_218/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_motiongate_218/test/ours_60000",
    ),
    "253_motiongate": dict(
        data_dir=f"{PROJECT}/data/253_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_253/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_motiongate_253/test/ours_60000",
    ),
    "302_motiongate": dict(
        data_dir=f"{PROJECT}/data/302_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/free60k_baseline_302/test/ours_60000",
        hair_dir=f"{PROJECT}/output/free60k_motiongate_302/test/ours_60000",
    ),
    # CSC dev-sweep: K=64 strand topology (vs default K=32 static-only baselines)
    "306_k64static": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/k64static_306/test/ours_60000",
    ),
    "302_k64static": dict(
        data_dir=f"{PROJECT}/data/UNION10_302_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_302/test/ours_60000",
        hair_dir=f"{PROJECT}/output/k64static_302/test/ours_60000",
    ),
    "074_k64static": dict(
        data_dir=f"{PROJECT}/data/UNION10_074_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/UNION10EMOEXP_074_eval_600k/test/ours_60000",
        hair_dir=f"{PROJECT}/output/k64static_074/test/ours_60000",
    ),
    # CSC dev-sweep: relaxed magnitude threshold/lambda (threshold_strand_static 0.3->0.6, lambda 1e-2->3e-3), K=32
    "306_relaxedstatic": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/relaxedstatic_306/test/ours_60000",
    ),
    "302_relaxedstatic": dict(
        data_dir=f"{PROJECT}/data/UNION10_302_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_302/test/ours_60000",
        hair_dir=f"{PROJECT}/output/relaxedstatic_302/test/ours_60000",
    ),
    "074_relaxedstatic": dict(
        data_dir=f"{PROJECT}/data/UNION10_074_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/UNION10EMOEXP_074_eval_600k/test/ours_60000",
        hair_dir=f"{PROJECT}/output/relaxedstatic_074/test/ours_60000",
    ),
    # Chain-propagated rotation (position chain + rotation chain) vs the adopted static-only
    # (position-only, K=32, threshold_strand_coherence=0.2) baseline, all 9 subjects.
    # 218 has no prior static-only run -> compared against the raw rigid baseline instead.
    "306_strandrot": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/cohthresh_02_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/strandrot_306/test/ours_60000",
    ),
    "104_strandrot": dict(
        data_dir=f"{PROJECT}/data/UNION10_104_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/cohthresh_02_104/test/ours_60000",
        hair_dir=f"{PROJECT}/output/strandrot_104/test/ours_60000",
    ),
    "253_strandrot": dict(
        data_dir=f"{PROJECT}/data/UNION10_253_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/cohthresh_02_253/test/ours_60000",
        hair_dir=f"{PROJECT}/output/strandrot_253/test/ours_60000",
    ),
    "460_strandrot": dict(
        data_dir=f"{PROJECT}/data/UNION10_460_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/cohthresh_02_460/test/ours_60000",
        hair_dir=f"{PROJECT}/output/strandrot_460/test/ours_60000",
    ),
    "264_strandrot": dict(
        data_dir=f"{PROJECT}/data/UNION10_264_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/cohthresh_02_264/test/ours_60000",
        hair_dir=f"{PROJECT}/output/strandrot_264/test/ours_60000",
    ),
    "302_strandrot": dict(
        data_dir=f"{PROJECT}/data/UNION10_302_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/cohthresh_02_302/test/ours_60000",
        hair_dir=f"{PROJECT}/output/strandrot_302/test/ours_60000",
    ),
    "304_strandrot": dict(
        data_dir=f"{PROJECT}/data/UNION10_304_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/cohthresh_02_304/test/ours_60000",
        hair_dir=f"{PROJECT}/output/strandrot_304/test/ours_60000",
    ),
    "074_strandrot": dict(
        data_dir=f"{PROJECT}/data/UNION10_074_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/staticonly60k_074/test/ours_60000",
        hair_dir=f"{PROJECT}/output/strandrot_074/test/ours_60000",
    ),
    "218_strandrot_vs_baseline": dict(
        data_dir=f"{PROJECT}/data/UNION10_218_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/UNION10EMOEXP_218_eval_600k/test/ours_60000",
        hair_dir=f"{PROJECT}/output/strandrot_218/test/ours_60000",
    ),
    # Periodic nearest-triangle rebinding only (no strand chain), vs rigid baseline -- tests
    # whether permanent binding itself is the bottleneck (TeGA's critique), independent of C1/C2.
    "306_rebind": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/fair60k_baseline_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/rebind_306/test/ours_60000",
    ),
    # K=16 (coarser than adopted K=32) vs the adopted K=32 static-only baseline
    "306_k16static": dict(
        data_dir=f"{PROJECT}/data/UNION10_306_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/cohthresh_02_306/test/ours_60000",
        hair_dir=f"{PROJECT}/output/k16static_306/test/ours_60000",
    ),
    "302_k16static": dict(
        data_dir=f"{PROJECT}/data/UNION10_302_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/cohthresh_02_302/test/ours_60000",
        hair_dir=f"{PROJECT}/output/k16static_302/test/ours_60000",
    ),
    "074_k16static": dict(
        data_dir=f"{PROJECT}/data/UNION10_074_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine",
        base_dir=f"{PROJECT}/output/UNION10EMOEXP_074_eval_600k/test/ours_60000",
        hair_dir=f"{PROJECT}/output/k16static_074/test/ours_60000",
    ),
}

flame = FlameHead(shape_params=300, expr_params=100).cuda()
hair_vid = flame.mask.v.hair.cpu().numpy()


def load(d, f):
    return np.asarray(Image.open(os.path.join(d, f)).convert("RGB"), dtype=np.float32) / 255.0


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else -10 * np.log10(mse)


def run_subject(sid, cfg):
    data_dir, base_dir, hair_dir = cfg["data_dir"], cfg["base_dir"], cfg["hair_dir"]
    with open(f"{data_dir}/transforms_test.json") as f:
        frames = json.load(f)["frames"]

    _vert_cache = {}

    def get_hair_verts_2d(frame):
        flame_path = os.path.join(data_dir, frame["flame_param_path"])
        if flame_path not in _vert_cache:
            d = dict(np.load(flame_path, allow_pickle=True))
            with torch.no_grad():
                out = flame(
                    torch.from_numpy(d["shape"]).float().cuda()[None, ...] if d["shape"].ndim == 1 else torch.from_numpy(d["shape"]).float().cuda(),
                    torch.from_numpy(d["expr"]).float().cuda(),
                    torch.from_numpy(d["rotation"]).float().cuda(),
                    torch.from_numpy(d["neck_pose"]).float().cuda(),
                    torch.from_numpy(d["jaw_pose"]).float().cuda(),
                    torch.from_numpy(d["eyes_pose"]).float().cuda(),
                    torch.from_numpy(d["translation"]).float().cuda(),
                    zero_centered_at_root_node=False, return_landmarks=False, return_verts_cano=False,
                    static_offset=torch.from_numpy(d["static_offset"]).float().cuda(),
                )
            verts = out[0] if isinstance(out, (list, tuple)) else out
            _vert_cache[flame_path] = verts[0].detach().cpu().numpy()
        verts = _vert_cache[flame_path]
        hair_pts_world = verts[hair_vid]

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        pts_h = np.concatenate([hair_pts_world, np.ones((hair_pts_world.shape[0], 1))], axis=1)
        pts_cam = (w2c @ pts_h.T).T[:, :3]

        fx, fy, cx, cy = frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]
        z = np.clip(pts_cam[:, 2], 1e-4, None)
        u = fx * pts_cam[:, 0] / z + cx
        v = fy * pts_cam[:, 1] / z + cy
        return u, v

    def bbox_for_frame(frame, w, h, pad_frac=0.15):
        u, v = get_hair_verts_2d(frame)
        valid = (u > -w) & (u < 2 * w) & (v > -h) & (v < 2 * h)
        if valid.sum() < 5:
            return 0, 0, w, int(h * 0.35)
        u, v = u[valid], v[valid]
        x0, x1 = u.min(), u.max()
        y0, y1 = v.min(), v.max()
        pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
        x0, x1 = max(0, x0 - pw), min(w, x1 + pw)
        y0, y1 = max(0, y0 - ph), min(h, y1 + ph)
        return int(x0), int(y0), int(x1), int(y1)

    records = []
    for i, frame in enumerate(frames):
        fname = f"{i:05d}.png"
        try:
            gt = load(f"{base_dir}/gt", fname)
            base = load(f"{base_dir}/renders", fname)
            hair = load(f"{hair_dir}/renders", fname)
        except FileNotFoundError:
            continue
        h, w = gt.shape[:2]
        x0, y0, x1, y1 = bbox_for_frame(frame, w, h)
        if x1 <= x0 or y1 <= y0:
            continue
        gt_c, base_c, hair_c = gt[y0:y1, x0:x1], base[y0:y1, x0:x1], hair[y0:y1, x0:x1]
        records.append({
            "idx": i, "fname": fname,
            "l1_base": float(np.abs(gt_c - base_c).mean()), "l1_hair": float(np.abs(gt_c - hair_c).mean()),
            "psnr_base": float(psnr(gt_c, base_c)), "psnr_hair": float(psnr(gt_c, hair_c)),
        })

    l1b = np.array([r["l1_base"] for r in records]); l1h = np.array([r["l1_hair"] for r in records])
    pb = np.array([r["psnr_base"] for r in records]); ph = np.array([r["psnr_hair"] for r in records])
    delta = ph - pb
    print(f"\n=== Subject {sid}: N={len(records)} frames ===")
    print(f"L1:   base={l1b.mean():.5f}  ours={l1h.mean():.5f}  ({(1 - l1h.mean()/l1b.mean())*100:+.2f}%)")
    print(f"PSNR: base={pb.mean():.3f}  ours={ph.mean():.3f}  ({delta.mean():+.3f} dB, std={delta.std():.3f})")
    print(f"Frames where ours better: {(delta>0).sum()}/{len(delta)} ({(delta>0).mean()*100:.1f}%)")

    with open(f"{PROJECT}/dev/hair_avatars/precise_hair_crop_{sid}.json", "w") as f:
        json.dump(records, f, indent=1)

    return dict(subject=sid, n=len(records), l1_base=float(l1b.mean()), l1_hair=float(l1h.mean()),
                psnr_base=float(pb.mean()), psnr_hair=float(ph.mean()), delta_mean=float(delta.mean()),
                delta_std=float(delta.std()), pct_better=float((delta>0).mean()*100))


if __name__ == "__main__":
    import sys
    subjects = sys.argv[1:] if len(sys.argv) > 1 else list(SUBJECTS.keys())
    summary = []
    for sid in subjects:
        summary.append(run_subject(sid, SUBJECTS[sid]))

    print("\n\n===== SUMMARY ACROSS SUBJECTS =====")
    for s in summary:
        print(f"{s['subject']}: PSNR delta={s['delta_mean']:+.3f}dB (std={s['delta_std']:.3f}), "
              f"L1 improvement={(1-s['l1_hair']/s['l1_base'])*100:+.2f}%, better-frames={s['pct_better']:.1f}%")

    with open(f"{PROJECT}/dev/hair_avatars/precise_hair_crop_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
