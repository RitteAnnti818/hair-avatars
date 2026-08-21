#!/bin/bash
set -uo pipefail
cd /home/hee/hee_data/GaussianAvatars
PY=/home/hee/miniconda3/envs/gaussian-avatars/bin/python
export CUDA_VISIBLE_DEVICES=7

echo "=== rendering 218 baseline at iteration 60000 (missing) ==="
$PY render.py -m output/UNION10EMOEXP_218_eval_600k --iteration 60000 --skip_train --skip_val \
  > render_218_baseline_60000.log 2>&1
echo "render 218 baseline exit $?"

echo "=== rendering remaining strandrot checkpoints (306, 104 already done) ==="
for subj in 253 460 264 302 304 074 218; do
  $PY render.py -m "output/strandrot_${subj}" --iteration 60000 --skip_train --skip_val \
    > "render_strandrot_${subj}.log" 2>&1
  echo "render strandrot_${subj} exit $?"
done

echo "=== metrics.py (all 9) ==="
$PY metrics.py -m output/strandrot_306 output/strandrot_104 output/strandrot_253 output/strandrot_460 \
                  output/strandrot_264 output/strandrot_302 output/strandrot_304 output/strandrot_074 output/strandrot_218 \
  > metrics_strandrot.log 2>&1
echo "metrics exit $?"

echo "=== precise hair-crop eval ==="
PYTHONPATH=. $PY dev/hair_avatars/precise_hair_crop_multi.py \
  306_strandrot 104_strandrot 253_strandrot 460_strandrot 264_strandrot 302_strandrot 304_strandrot \
  074_strandrot 218_strandrot_vs_baseline \
  > precise_hair_crop_strandrot.log 2>&1
echo "crop-eval exit $?"

echo "=== STRANDROT FULL EVAL COMPLETE ==="
