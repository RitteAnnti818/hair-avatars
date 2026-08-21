#!/bin/bash
set -uo pipefail
cd /home/hee/hee_data/GaussianAvatars
PY=/home/hee/miniconda3/envs/gaussian-avatars/bin/python
export CUDA_VISIBLE_DEVICES=6

# B-2a sanity: static-only inextensible (hard-clamp) chain vs. the existing free-vector
# static-only FREE baseline (306: +0.281dB, 302: +0.014dB precise-hair-crop dPSNR).
# --disable_strand_dynamic keeps this apples-to-apples with the static-only comparison point.

declare -A PORTS=( [306]=63320 [302]=63321 )
for subj in 306 302; do
  port="${PORTS[$subj]}"
  echo "=== starting inext (static-only) training for subject $subj (port $port) ==="
  $PY train.py \
    -s "data/${subj}_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine" \
    -m "output/inext60k_${subj}_FREE" \
    --eval --white_background --bind_to_mesh \
    --enable_hair_strands --strand_json_path dev/hair_avatars/phase1_strands_k32.json \
    --enable_inextensible_chain --disable_strand_dynamic \
    --iterations 60000 --port "$port" \
    > "train_inext60k_${subj}_FREE.log" 2>&1
  echo "=== finished subject $subj (exit $?) ==="
done
echo "=== B-2a TRAINING DONE, rendering + metrics ==="

for subj in 306 302; do
  $PY render.py -m "output/inext60k_${subj}_FREE" --iteration 60000 --skip_train --skip_val \
    > "render_inext60k_${subj}_FREE.log" 2>&1
  echo "render $subj exit $?"
done

$PY metrics.py -m output/inext60k_306_FREE output/inext60k_302_FREE \
  > metrics_inext_b2a.log 2>&1
echo "metrics exit $?"

echo "=== B-2a FULL SANITY RUN COMPLETE ==="
