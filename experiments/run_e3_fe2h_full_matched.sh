#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
REPO_DIR="${REPO_DIR:-/root/vpsc}"
RESULT_DIR="${RESULT_DIR:-${REPO_DIR}/results/e3_scan}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RESULT_DIR}/checkpoints}"
SHARED_INIT="${SHARED_INIT:-${CHECKPOINT_DIR}/e3_fe2h_full_shared_init.pt}"

mkdir -p "${RESULT_DIR}" "${CHECKPOINT_DIR}"
cd "${REPO_DIR}"

exec 9>"${RESULT_DIR}/.e3_fe2h_full_suite.lock"
if ! flock -n 9; then
  echo "REFUSE another full matched suite already holds the lock" >&2
  exit 1
fi

is_complete() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
try:
    artifact = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
gates = artifact.get("matched_gates") or {}
complete = (
    artifact.get("status") == "COMPLETED"
    and gates.get("training_complete") is True
    and gates.get("validation_complete") is True
)
raise SystemExit(0 if complete else 1)
PY
}

run_variant() {
  local label="$1"
  local tile_size="$2"
  local active_tiles="$3"
  local create_init="$4"
  local artifact="${RESULT_DIR}/e3_fe2h_full_${label}.json"
  local checkpoint="${CHECKPOINT_DIR}/e3_fe2h_full_${label}.pt"
  local log="${RESULT_DIR}/e3_fe2h_full_${label}.log"
  local progress_log="${RESULT_DIR}/e3_fe2h_full_${label}.tqdm.log"

  if is_complete "${artifact}"; then
    echo "SKIP completed ${label}"
    return
  fi

  local interrupted_stamp
  interrupted_stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
  local stale_path
  for stale_path in "${artifact}" "${checkpoint}" "${log}" "${progress_log}"; do
    if [[ -e "${stale_path}" ]]; then
      local archived_path="${stale_path%.*}_incomplete_${interrupted_stamp}.${stale_path##*.}"
      mv "${stale_path}" "${archived_path}"
      echo "ARCHIVE incomplete ${stale_path} -> ${archived_path}"
    fi
  done

  local create_args=()
  if [[ "${create_init}" == "yes" ]]; then
    create_args+=(--create-shared-init)
  fi

  echo "START ${label} $(date -Iseconds)"
  "${PYTHON_BIN}" experiments/e3_fe2h_scale_train.py \
    --label "full_${label}" \
    --out "${artifact}" \
    --checkpoint "${checkpoint}" \
    --save-optimizer \
    --shared-init "${SHARED_INIT}" \
    "${create_args[@]}" \
    --d-model 8192 \
    --state-dim 8192 \
    --tile-size "${tile_size}" \
    --active-tiles "${active_tiles}" \
    --block-size 32 \
    --rank 512 \
    --batch-size 112 \
    --seq-len 128 \
    --train-mode full_epoch \
    --epochs 1 \
    --full-validation \
    --warmup-steps 10 \
    --log-every 10 \
    --finite-check-every 25 \
    --learning-rate 3e-4 \
    --weight-decay 0.01 \
    --route-supervision-weight 0.01 \
    --homeostasis-weight 1.0 \
    --amp-init-scale 256 \
    --amp-growth-interval 100000 \
    --checkpoint-every-steps 1000 \
    --tqdm-progress \
    --seed 0 \
    --sample-interval-ms 200 \
    >"${log}" 2>"${progress_log}"
  echo "DONE ${label} $(date -Iseconds)"
}

run_variant coarse_k2 2048 2 yes
run_variant micro_k16 256 16 no
run_variant micro_k8 256 8 no
run_variant micro_k4 256 4 no

echo "SUITE_DONE $(date -Iseconds)"
