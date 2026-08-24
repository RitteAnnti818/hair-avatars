#!/bin/bash
set -uo pipefail
cd /home/hee/hee_data/GaussianAvatars
PY=/home/hee/miniconda3/envs/gaussian-avatars/bin/python
export CUDA_VISIBLE_DEVICES=7

echo "=== waiting for round-2 (304, 460) to finish ==="
while ! grep -q "Training complete\." train_free60k_motiongate_460.log 2>/dev/null; do
  sleep 30
done
echo "=== round-2 confirmed done, starting round-3 (074,104,218,253,302) ==="

declare -A PORTS=( [074]=63305 [104]=63306 [218]=63307 [253]=63308 [302]=63309 )
for subj in 074 104 218 253 302; do
  port="${PORTS[$subj]}"
  echo "=== starting motion-gate training for subject $subj (port $port) ==="
  $PY train.py \
    -s "data/${subj}_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine" \
    -m "output/free60k_motiongate_${subj}" \
    --eval --white_background --bind_to_mesh \
    --enable_hair_strands --strand_json_path dev/hair_avatars/phase1_strands_k32.json \
    --enable_motion_gate --motion_gate_percentile 90.0 \
    --iterations 60000 --port "$port" \
    > "train_free60k_motiongate_${subj}.log" 2>&1
  echo "=== finished subject $subj (exit $?) ==="
done
echo "=== round-3 training done ==="

echo "=== rendering + metrics for 074,104,218,253,302 ==="
for subj in 074 104 218 253 302; do
  $PY render.py -m "output/free60k_motiongate_${subj}" --iteration 60000 --skip_train --skip_val \
    > "render_motiongate_${subj}.log" 2>&1
  echo "render $subj exit $?"
done
$PY metrics.py -m output/free60k_motiongate_074 output/free60k_motiongate_104 output/free60k_motiongate_218 \
                  output/free60k_motiongate_253 output/free60k_motiongate_302 \
  > metrics_motiongate_round3.log 2>&1
echo "metrics round3 exit $?"

echo "=== precise hair-crop eval, ALL 9 motiongate subjects ==="
PYTHONPATH=. $PY dev/hair_avatars/precise_hair_crop_multi.py \
  264_motiongate 306_motiongate 304_motiongate 460_motiongate \
  074_motiongate 104_motiongate 218_motiongate 253_motiongate 302_motiongate \
  > precise_hair_crop_motiongate_ALL9.log 2>&1
echo "crop-eval exit $?"

echo "=== ALL MOTION-GATE WORK (9/9 SUBJECTS) COMPLETE ==="
