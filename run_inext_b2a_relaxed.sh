#!/bin/bash
set -uo pipefail
cd /home/hee/hee_data/GaussianAvatars
PY=/home/hee/miniconda3/envs/gaussian-avatars/bin/python
export CUDA_VISIBLE_DEVICES=6

# B-2a retry: relax the hard-clamp cap (and threshold_strand_static in lockstep, so L_static stays
# a clean ~0 sanity signal) from 0.3 -> 0.5, testing whether cap==0.3 was simply too tight relative
# to what the soft-penalty free-vector model was actually using (which can exceed 0.3 at a small
# linear cost). Static-only (--disable_strand_dynamic), same two subjects (306, 302), same GPU.

declare -A PORTS=( [306]=63322 [302]=63323 )
for subj in 306 302; do
  port="${PORTS[$subj]}"
  echo "=== [relaxed cap=0.5] launching subject $subj (port $port) ==="
  $PY train.py \
    -s "data/${subj}_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine" \
    -m "output/inext60k_relaxed_${subj}_FREE" \
    --eval --white_background --bind_to_mesh \
    --enable_hair_strands --strand_json_path dev/hair_avatars/phase1_strands_k32.json \
    --enable_inextensible_chain --disable_strand_dynamic \
    --inextensible_static_cap 0.5 --threshold_strand_static 0.5 \
    --iterations 60000 --port "$port" \
    > "train_inext60k_relaxed_${subj}_FREE.log" 2>&1 &
done
wait
echo "=== both relaxed-cap trainings complete, rendering ==="

for subj in 306 302; do
  $PY render.py -m "output/inext60k_relaxed_${subj}_FREE" --iteration 60000 --skip_train --skip_val \
    > "render_inext60k_relaxed_${subj}_FREE.log" 2>&1
  echo "render $subj exit $?"
done

echo "=== precise hair-crop eval (relaxed cap) ==="
PYTHONPATH=. $PY dev/hair_avatars/precise_hair_crop_multi.py 306_inext_relaxed 302_inext_relaxed \
  > precise_hair_crop_inext_relaxed.log 2>&1
echo "crop-eval exit $?"

echo "=== B-2a RELAXED-CAP RUN COMPLETE ==="
