#!/bin/bash
set -uo pipefail
cd /home/hee/hee_data/GaussianAvatars
PY=/home/hee/miniconda3/envs/gaussian-avatars/bin/python
export CUDA_VISIBLE_DEVICES=7

echo "=== waiting for k64static_306 (currently running via sequential wrapper) to finish ==="
while ! grep -q "Training complete\." train_k64static_306.log 2>/dev/null; do
  sleep 5
done
echo "=== k64static_306 done. Stopping sequential wrapper before it launches job 2 ==="
pkill -f "run_csc_devsweep.sh" 2>/dev/null
sleep 3
if pgrep -f "output/k64static_302" > /dev/null; then
  echo "WARNING: k64static_302 was already started by the wrapper before kill -- killing it to avoid a duplicate/conflicting run"
  pkill -f "output/k64static_302"
  sleep 3
fi

echo "=== launching remaining 5 jobs in parallel (GPU 7 only) ==="
run_train() {
  local subj="$1" tag="$2" strand_json="$3" thr_static="$4" lam_static="$5" port="$6"
  $PY train.py \
    -s "data/UNION10_${subj}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine" \
    -m "output/${tag}_${subj}" \
    --eval --white_background --bind_to_mesh \
    --enable_hair_strands --strand_json_path "$strand_json" --disable_strand_dynamic \
    --threshold_strand_coherence 0.2 \
    --threshold_strand_static "$thr_static" --lambda_strand_static "$lam_static" \
    --iterations 60000 --port "$port" \
    > "train_${tag}_${subj}.log" 2>&1
  echo "=== finished ${tag}_${subj} exit $? ==="
}

run_train 302 k64static      "dev/hair_avatars/phase1_strands_k64.json" 0.3 1e-2 63402 &
run_train 074 k64static      "dev/hair_avatars/phase1_strands_k64.json" 0.3 1e-2 63403 &
run_train 306 relaxedstatic  "dev/hair_avatars/phase1_strands_k32.json" 0.6 3e-3 63404 &
run_train 302 relaxedstatic  "dev/hair_avatars/phase1_strands_k32.json" 0.6 3e-3 63405 &
run_train 074 relaxedstatic  "dev/hair_avatars/phase1_strands_k32.json" 0.6 3e-3 63406 &
wait
echo "=== ALL 5 PARALLEL JOBS DONE ==="

echo "=== rendering + metrics ==="
for tag in k64static relaxedstatic; do
  for subj in 306 302 074; do
    $PY render.py -m "output/${tag}_${subj}" --iteration 60000 --skip_train --skip_val > "render_${tag}_${subj}.log" 2>&1
    echo "render ${tag}_${subj} exit $?"
  done
done
$PY metrics.py -m output/k64static_306 output/k64static_302 output/k64static_074 \
                  output/relaxedstatic_306 output/relaxedstatic_302 output/relaxedstatic_074 \
  > metrics_csc_devsweep.log 2>&1
echo "metrics exit $?"

echo "=== CSC DEV-SWEEP (K + THRESHOLD, parallel) COMPLETE ==="
