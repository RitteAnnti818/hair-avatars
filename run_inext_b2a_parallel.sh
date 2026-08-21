#!/bin/bash
set -uo pipefail
cd /home/hee/hee_data/GaussianAvatars
PY=/home/hee/miniconda3/envs/gaussian-avatars/bin/python
export CUDA_VISIBLE_DEVICES=6

# 306 is already training independently (launched earlier, PID reparented to init after the
# original sequential wrapper was killed). This script launches 302 alongside it in parallel,
# then waits on each training log independently and does render+metrics per subject as soon as
# it finishes (not waiting for the other), finally running the combined 2-subject metrics.py.

echo "=== launching 302 (parallel with already-running 306) ==="
$PY train.py \
  -s "data/302_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine" \
  -m "output/inext60k_302_FREE" \
  --eval --white_background --bind_to_mesh \
  --enable_hair_strands --strand_json_path dev/hair_avatars/phase1_strands_k32.json \
  --enable_inextensible_chain --disable_strand_dynamic \
  --iterations 60000 --port 63321 \
  > "train_inext60k_302_FREE.log" 2>&1 &
TRAIN_302_PID=$!
echo "302 training PID: $TRAIN_302_PID"

render_and_eval() {
  local subj=$1
  echo "=== [$subj] training complete, rendering ==="
  $PY render.py -m "output/inext60k_${subj}_FREE" --iteration 60000 --skip_train --skip_val \
    > "render_inext60k_${subj}_FREE.log" 2>&1
  echo "=== [$subj] render exit $? ==="
}

# poll 306 (already running before this script started)
while ! grep -q "Training complete\." train_inext60k_306_FREE.log 2>/dev/null; do
  sleep 30
done
render_and_eval 306

# poll 302 (launched by this script)
while ! grep -q "Training complete\." train_inext60k_302_FREE.log 2>/dev/null; do
  sleep 30
done
render_and_eval 302

echo "=== both subjects rendered, running combined metrics.py ==="
$PY metrics.py -m output/inext60k_306_FREE output/inext60k_302_FREE \
  > metrics_inext_b2a.log 2>&1
echo "metrics exit $?"

echo "=== B-2a PARALLEL SANITY RUN COMPLETE ==="
