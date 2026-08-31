#!/bin/bash
set -euo pipefail

MODE="${1:-all}"
SUBJECT="${SUBJECT:-306}"
GPU="${GPU:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STRAND_JSON="${STRAND_JSON:-dev/hair_avatars/phase1_strands_k32.json}"
PORT_A="${PORT_A:-60000}"
PORT_B="${PORT_B:-60001}"
PORT_C="${PORT_C:-60002}"
PORT_D="${PORT_D:-60003}"
PORT_E="${PORT_E:-60004}"
PORT_F="${PORT_F:-60005}"

DATA_PATH="data/UNION10_${SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine"
BASE_MODEL="output/UNION10EMOEXP_${SUBJECT}_eval_600k"

run_cmd() {
  local tag="$1"
  local port="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" train.py \
    -s "$DATA_PATH" \
    -m "${BASE_MODEL}_${tag}" \
    --eval --bind_to_mesh --white_background --port "$port" \
    --enable_hair_strands --strand_json_path "$STRAND_JSON" \
    --iterations 60000 \
    "$@"
}

case "$MODE" in
  a|A|smooth)
    run_cmd A "$PORT_A" \
      --strand_temporal_mode smooth \
      --lambda_strand_temporal_smooth 1e-3
    ;;
  b|B|pose_gate)
    run_cmd B "$PORT_B" \
      --strand_temporal_mode pose_gate \
      --enable_motion_gate \
      --motion_gate_percentile 90 \
      --lambda_strand_temporal_gate_l2 1e-4
    ;;
  c|C|strand_gate)
    run_cmd C "$PORT_C" \
      --strand_temporal_mode strand_gate \
      --enable_motion_gate \
      --motion_gate_percentile 90 \
      --lambda_strand_temporal_gate_l2 1e-4
    ;;
  d|D|dynamic_lr)
    run_cmd D "$PORT_D" \
      --strand_dynamic_lr 3e-4
    ;;
  e|E|warmup)
    run_cmd E "$PORT_E" \
      --strand_dynamic_warmup_iters 10000
    ;;
  f|F|tip)
    run_cmd F "$PORT_F" \
      --strand_dynamic_tip_power 2.0
    ;;
  all)
    "$0" A
    "$0" B
    "$0" C
    ;;
  extended)
    "$0" D
    "$0" E
    "$0" F
    ;;
  *)
    echo "Usage: $0 [A|B|C|D|E|F|all|extended]"
    echo "Optional env: SUBJECT=306 GPU=2 PYTHON_BIN=python STRAND_JSON=... PORT_A=60000 PORT_B=60001 PORT_C=60002 PORT_D=60003 PORT_E=60004 PORT_F=60005"
    exit 1
    ;;
esac
