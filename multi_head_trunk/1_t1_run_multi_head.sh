#!/usr/bin/env bash

set -e

gpu=0
PYTHON_BIN="${PYTHON_BIN:-python3}"
config="multi_head_trunk/conf/xlsr_3_11_24_multi_head.yaml"

RUN_TRAIN=1
RUN_DEV_ANALYZE=1
RUN_SCORE=1

wandb_mode=offline
wandb_project="3090-Track1"
wandb_entity=""
wandb_run_name=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT}" || exit 1

WB_ARGS=(--wandb_mode "${wandb_mode}" --wandb_project "${wandb_project}")
[[ -n "${wandb_entity}" ]] && WB_ARGS+=(--wandb_entity "${wandb_entity}")
[[ -n "${wandb_run_name}" ]] && WB_ARGS+=(--wandb_run_name "${wandb_run_name}")

model_path="$("${PYTHON_BIN}" -c "
import yaml
with open('${config}', encoding='utf-8') as f:
    print(yaml.safe_load(f).get('out_fold'))
")"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONWARNINGS="ignore" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/main_train.py" \
      --config "${config}" \
      --gpu "${gpu}" \
      "${WB_ARGS[@]}"
fi

if [[ "${RUN_DEV_ANALYZE}" == "1" ]]; then
  PYTHONWARNINGS="ignore" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze.py" \
      --model_path "${model_path}" \
      --gpu "${gpu}" \
      --batch_size 160 \
      --eval_task atadd-track1 \
      --metrics_only
fi

if [[ "${RUN_SCORE}" == "1" ]]; then
  PYTHONWARNINGS="ignore" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/inference.py" \
      --model_path "${model_path}" \
      --gpu "${gpu}" \
      --batch_size 160 \
      --eval_task atadd-track1 \
      --threshold 0.5
fi
