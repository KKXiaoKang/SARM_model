#!/usr/bin/env bash
# ============================================================================
# SARM 多卡训练（accelerate launch + DDP）
# ============================================================================
# 依赖（训练 SARM 前必装，否则会报 ImportError: faker）:
#   pip install 'lerobot[training]' 'lerobot[sarm]'
#   # 或: pip install faker
# 若 WANDB_ENABLE=true（默认见下方变量），还需 wandb:
#   pip install wandb
#   # lerobot[training] 已包含 wandb
#
# 参考 train_groot_multi_gpu.sh：通过 --gpu / CUDA_VISIBLE_DEVICES 选卡并自动设置进程数。
#
# 用法:
#   ./examples/training/train_sarm_multi_gpu.sh
#   ./examples/training/train_sarm_multi_gpu.sh --gpu 0,1,2,3
#   ./examples/training/train_sarm_multi_gpu.sh -g 0,1
#
# 环境变量（可选）:
#   DATASET_REPO_ID   本地数据集路径或 Hub id
#   GPU_IDS            与 --gpu 相同效果（逗号分隔物理 GPU id）
#   NUM_PROCESSES      仅在不指定 GPU 时使用；否则由 GPU 个数自动推断
#   MIXED_PRECISION    传给 accelerate，默认 bf16；设为 no 可关闭
#
# 其余参数会原样传给 lerobot-train，例如:
#   ./examples/training/train_sarm_multi_gpu.sh --gpu 0,1 --seed=42 --log_freq=100
#
# 说明:
#   - batch_size 为每进程 batch；有效 batch ≈ batch_size × 进程数（见训练日志）。
#   - LeRobot 不会自动按卡数缩放学习率；若 loss 不稳可自行调 policy 学习率或 batch。
# ============================================================================

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

# --- 解析本脚本专有参数；其余留给 lerobot-train ---
GPU_IDS_CLI=""
EXTRA_TRAIN_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu|-g)
      GPU_IDS_CLI="$2"
      shift 2
      ;;
    *)
      EXTRA_TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

# --- 默认训练超参（可用环境变量覆盖）---
: "${DATASET_REPO_ID:=/home/kangkk2/lerobot/lerobot_dataset/0415_pick_cube_single_s62_real_clawAsS-filtered}"
: "${DATASET_ROOT:=}"
: "${POLICY_IMAGE_KEY:=observation.images.cam_head}"
: "${ANNOTATION_MODE:=dual}"
: "${OUTPUT_DIR:=outputs/train/sarm_dual_$(date +%Y%m%d_%H%M%S)}"
: "${BATCH_SIZE:=32}"
: "${STEPS:=5000}"
: "${WANDB_ENABLE:=true}"
: "${WANDB_PROJECT:=sarm}"
: "${PUSH_TO_HUB:=false}"
: "${POLICY_REPO_ID:=your-username/your-sarm-model}"
: "${MIXED_PRECISION:=bf16}"

# --- GPU：优先命令行 --gpu，其次环境变量 GPU_IDS；否则用可见 GPU 数量 ---
if [[ -n "${GPU_IDS_CLI}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS_CLI}"
  NUM_PROCESSES=$(echo "${GPU_IDS_CLI}" | tr ',' '\n' | sed '/^$/d' | wc -l)
elif [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  NUM_PROCESSES=$(echo "${GPU_IDS}" | tr ',' '\n' | sed '/^$/d' | wc -l)
else
  : "${NUM_PROCESSES:=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 4)}"
fi

if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "error: no GPU available (NUM_PROCESSES=${NUM_PROCESSES})" >&2
  exit 1
fi

LEROBOT_TRAIN="$(command -v lerobot-train || true)"
if [[ -z "$LEROBOT_TRAIN" ]]; then
  echo "error: lerobot-train not found on PATH. Install with: pip install 'lerobot[training]'" >&2
  exit 1
fi

TRAIN_ARGS=(
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--policy.type=sarm"
  "--policy.annotation_mode=${ANNOTATION_MODE}"
  "--policy.image_key=${POLICY_IMAGE_KEY}"
  "--output_dir=${OUTPUT_DIR}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--wandb.enable=${WANDB_ENABLE}"
  "--wandb.project=${WANDB_PROJECT}"
  "--policy.push_to_hub=${PUSH_TO_HUB}"
)

if [[ -n "${DATASET_ROOT}" ]]; then
  TRAIN_ARGS+=("--dataset.root=${DATASET_ROOT}")
fi

if [[ "${PUSH_TO_HUB}" == "true" ]]; then
  TRAIN_ARGS+=("--policy.repo_id=${POLICY_REPO_ID}")
fi

echo "=========================================="
echo "SARM multi-GPU training"
echo "  num_processes=${NUM_PROCESSES}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
echo "  mixed_precision=${MIXED_PRECISION}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  dataset.repo_id=${DATASET_REPO_ID}"
echo "=========================================="

ACCELERATE_ARGS=(
  --multi_gpu
  --num_processes="${NUM_PROCESSES}"
  --mixed_precision="${MIXED_PRECISION}"
)

exec accelerate launch \
  "${ACCELERATE_ARGS[@]}" \
  "${LEROBOT_TRAIN}" \
  "${TRAIN_ARGS[@]}" \
  "${EXTRA_TRAIN_ARGS[@]}"
