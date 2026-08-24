#!/bin/bash
set -uo pipefail
cd /home/hee/hee_data/GaussianAvatars
PY=/home/hee/miniconda3/envs/gaussian-avatars/bin/python
export CUDA_VISIBLE_DEVICES=7

declare -A PORTS=( [306_k64]=63401 [302_k64]=63402 [074_k64]=63403 [306_relax]=63404 [302_relax]=63405 [074_relax]=63406 )

run_train() {
  local subj="$1" tag="$2" strand_json="$3" thr_static="$4" lam_static="$5" port="$6"
  echo "=== starting $tag ($subj, port $port) ==="
  $PY train.py \
    -s "data/UNION10_${subj}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine" \
    -m "output/${tag}_${subj}" \
    --eval --white_background --bind_to_mesh \
    --enable_hair_strands --strand_json_path "$strand_json" --disable_strand_dynamic \
    --threshold_strand_coherence 0.2 \
    --threshold_strand_static "$thr_static" --lambda_strand_static "$lam_static" \
    --iterations 60000 --port "$port" \
    > "train_${tag}_${subj}.log" 2>&1
  echo "=== finished $tag ($subj) exit $? ==="
}

# K=64 sweep (default threshold_strand_static=0.3, lambda_strand_static=1e-2)
for subj in 306 302 074; do
  run_train "$subj" "k64static" "dev/hair_avatars/phase1_strands_k64.json" 0.3 1e-2 "${PORTS[${subj}_k64]}"
done

# relaxed magnitude threshold sweep (default K=32 topology, loosened threshold/lambda)
for subj in 306 302 074; do
  run_train "$subj" "relaxedstatic" "dev/hair_avatars/phase1_strands_k32.json" 0.6 3e-3 "${PORTS[${subj}_relax]}"
done

echo "=== all 6 CSC dev-sweep training runs done, rendering+metrics ==="
for tag in k64static relaxedstatic; do
  for subj in 306 302 074; do
    $PY render.py -m "output/${tag}_${subj}" --iteration 60000 --skip_train --skip_val \
      > "render_${tag}_${subj}.log" 2>&1
    echo "render ${tag}_${subj} exit $?"
  done
done
$PY metrics.py -m output/k64static_306 output/k64static_302 output/k64static_074 \
                  output/relaxedstatic_306 output/relaxedstatic_302 output/relaxedstatic_074 \
  > metrics_csc_devsweep.log 2>&1
echo "metrics exit $?"

echo "=== CSC DEV-SWEEP (K + THRESHOLD) COMPLETE ==="
