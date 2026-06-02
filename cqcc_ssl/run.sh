cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

gpu=0
config=cqcc_ssl/conf/base.yaml
model_path=""            # Leave empty to derive from config out_fold.
RESUME=0                 # 1 = resume from checkpoint
RUN_TRAIN=1              # 1 = train
RUN_SCORE=1              # 1 = generate eval scores
RUN_DEV_ANALYZE=0        # Reserved for parity with root run.sh

wandb_mode=offline
wandb_project="AT-ADD-CQCC-SSL"
wandb_entity=""
wandb_run_name=""

if [[ -z "${model_path}" && -n "${config}" ]]; then
    model_path=$(python3 -c "
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
    python3 cqcc_ssl/train.py \
        --resume "${model_path}" \
        --gpu    "${gpu}" \
        "${WB_ARGS[@]}"
elif [[ "${RUN_TRAIN}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 1] Training"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONWARNINGS="ignore" \
    python3 cqcc_ssl/train.py \
        --config "${config}" \
        --gpu    "${gpu}" \
        "${WB_ARGS[@]}"
fi

if [[ "${RUN_SCORE}" == "1" ]]; then
    echo ""
    echo ">>> [Stage 2] Score generation (eval set)"
    PYTHONWARNINGS="ignore" \
    python3 cqcc_ssl/inference.py \
        --model_path "${model_path}" \
        --gpu        "${gpu}" \
        --batch_size 160 \
        --threshold  0.5
fi

if [[ "${RUN_DEV_ANALYZE}" == "1" ]]; then
    echo "[warn] cqcc_ssl/run.sh keeps RUN_DEV_ANALYZE for parity; use root scripts/analyze.py only after registering this model."
fi
