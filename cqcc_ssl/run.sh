#!/usr/bin/env bash
# Same style as ThreeTO_track1/run.sh; safe to run as ./run.sh from cqcc_ssl/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}" || exit 1

# Prefer the project conda env (./run.sh does not always inherit ``conda activate``).
CONDA_ENV="${CONDA_ENV:-atadd_t1_3.10}"
if [[ -n "${CONDA_PREFIX}" && "$(basename "${CONDA_PREFIX}")" == "${CONDA_ENV}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python" ]]; then
    PYTHON="${HOME}/miniconda3/envs/${CONDA_ENV}/bin/python"
else
    PYTHON="python"
fi

if ! "${PYTHON}" -c "import torch, torchaudio" 2>/dev/null; then
    echo "[error] ${PYTHON} cannot import torch/torchaudio."
    echo "        Use: conda activate ${CONDA_ENV}"
    "${PYTHON}" -c "import sys; print('executable:', sys.executable)" 2>/dev/null || true
    exit 1
fi

gpu=0
config=cqcc_ssl/conf/base.yaml
model_path=""            # 留空则自动从 config 的 out_fold 读取
RESUME=0                 # 1 = 从 checkpoint 继续训练（自动加载 model_path/config.yaml）
RUN_TRAIN=1              # 1 = 训练
RUN_SCORE=1              # 1 = 生成 eval 集预测分数（inference）
RUN_DEV_ANALYZE=0        # 1 = dev 集分析

wandb_mode=offline
wandb_project="AT-ADD-CQCC-SSL"
wandb_entity=""
wandb_run_name=""

if [[ -z "${model_path}" && -n "${config}" ]]; then
    model_path=$("${PYTHON}" -c "
import yaml
with open('${config}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('out_fold', './ckpt_t1_cqcc_ssl/cross_attn'))
" 2>/dev/null)
fi
model_path=${model_path:-"./ckpt_t1_cqcc_ssl/cross_attn"}

echo "============================================================"
echo " Config     : ${config}"
echo " Model path : ${model_path}"
echo " Python     : $("${PYTHON}" -c 'import sys; print(sys.executable)')"
echo " GPU        : ${gpu}"
echo " W&B mode   : ${wandb_mode}"
echo "============================================================"

WB_ARGS=(--wandb_mode "${wandb_mode}" --wandb_project "${wandb_project}")
[[ -n "${wandb_entity}"   ]] && WB_ARGS+=(--wandb_entity "${wandb_entity}")
[[ -n "${wandb_run_name}" ]] && WB_ARGS+=(--wandb_run_name "${wandb_run_name}")

if [[ "${RESUME}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 1] Resume training from ${model_path}"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONWARNINGS="ignore" \
    "${PYTHON}" cqcc_ssl/train.py \
        --resume "${model_path}" \
        --gpu    "${gpu}" \
        "${WB_ARGS[@]}"
elif [[ "${RUN_TRAIN}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 1] Training"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONWARNINGS="ignore" \
    "${PYTHON}" cqcc_ssl/train.py \
        --config "${config}" \
        --gpu    "${gpu}" \
        "${WB_ARGS[@]}"
fi

if [[ "${RUN_SCORE}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 2] Score generation (eval set)"
    PYTHONWARNINGS="ignore" \
    "${PYTHON}" cqcc_ssl/inference.py \
        --model_path "${model_path}" \
        --gpu        "${gpu}" \
        --batch_size 160 \
        --threshold  0.5
fi

if [[ "${RUN_DEV_ANALYZE}" == "1" ]]; then
    echo ""
    echo "[warn] cqcc_ssl/run.sh keeps RUN_DEV_ANALYZE for parity;"
    echo "       use root scripts/analyze.py after registering this model."
fi
