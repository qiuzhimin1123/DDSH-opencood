#!/usr/bin/env bash
# Run DDSH-VoxelNeXt stages sequentially.
#
# Default workflow for each stage:
#   debug forward -> train -> global-sort inference -> DDSH statistics profile
#
# Outputs:
#   opencood/logs/ddsh_batch_runs/<timestamp>/manifest.csv
#   opencood/logs/ddsh_batch_runs/<timestamp>/*_{debug,train,infer,profile}.log
#   opencood/logs/<RUN_DIR>/eval*.yaml
#   opencood/logs/<RUN_DIR>/ddsh_statistics/*.csv

set -euo pipefail

DATASET="v2v4real"
CONDA_ENV="opencood"
USE_CONDA=1
RUN_DEBUG=1
RUN_INFER=1
RUN_PROFILE=1
PROFILE_MAX_BATCHES="-1"
PROFILE_SPLIT="val"
PROFILE_NUM_WORKERS="0"
PROFILE_BATCH_SIZE="1"
NO_DATA_CHECK=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_ddsh_all_stages.sh [options]

Options:
  --dataset v2v4real|opv2v     Dataset YAML group to run. Default: v2v4real.
  --conda-env NAME             Conda env to activate. Default: opencood.
  --no-conda                   Do not activate conda inside the script.
  --skip-debug                 Skip debug_ddsh_forward.py before training.
  --skip-infer                 Skip post-training inference.py.
  --skip-profile               Skip profile_ddsh_statistics.py.
  --profile-max-batches N      Profile first N batches; -1 means full split.
                               Default: -1.
  --profile-num-workers N      Dataloader workers for profile. Default: 0.
  --profile-batch-size N       Batch size for profile. Default: 1.
  --no-data-check              Do not check train/validate data folders.
  -h, --help                   Show this help.

Examples:
  bash scripts/run_ddsh_all_stages.sh

  bash scripts/run_ddsh_all_stages.sh --dataset v2v4real --profile-max-batches 100

  CUDA_VISIBLE_DEVICES=0 bash scripts/run_ddsh_all_stages.sh --skip-debug
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="$2"
      shift 2
      ;;
    --no-conda)
      USE_CONDA=0
      shift
      ;;
    --skip-debug)
      RUN_DEBUG=0
      shift
      ;;
    --skip-infer)
      RUN_INFER=0
      shift
      ;;
    --skip-profile)
      RUN_PROFILE=0
      shift
      ;;
    --profile-max-batches)
      PROFILE_MAX_BATCHES="$2"
      shift 2
      ;;
    --profile-num-workers)
      PROFILE_NUM_WORKERS="$2"
      shift 2
      ;;
    --profile-batch-size)
      PROFILE_BATCH_SIZE="$2"
      shift 2
      ;;
    --no-data-check)
      NO_DATA_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

if [[ "$USE_CONDA" == "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    echo "Warning: conda was not found. Continuing with current Python."
  fi
fi

if [[ "$DATASET" == "v2v4real" ]]; then
  TRAIN_DIR="v2v4real/train"
  VAL_DIR="v2v4real/validate"
  YAMLS=(
    "opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage0_sparse_single.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage3_demand_supply.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage1_sparse_all_token.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage2_sparse_topk.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage4_sparse_attention.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_v2v4real_stage5_hybrid_late.yaml"
  )
elif [[ "$DATASET" == "opv2v" ]]; then
  TRAIN_DIR="opv2v_data_dumping/train"
  VAL_DIR="opv2v_data_dumping/validate"
  YAMLS=(
    "opencood/hypes_yaml/ddsh_voxelnext_stage0_sparse_single.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_stage3_demand_supply.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_stage1_sparse_all_token.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_stage2_sparse_topk.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_stage4_sparse_attention.yaml"
    "opencood/hypes_yaml/ddsh_voxelnext_stage5_hybrid_late.yaml"
  )
else
  echo "Unsupported dataset: $DATASET" >&2
  exit 2
fi

if [[ "$NO_DATA_CHECK" == "0" ]]; then
  if [[ ! -d "$TRAIN_DIR" || ! -d "$VAL_DIR" ]]; then
    echo "Data folders are missing for dataset=$DATASET:" >&2
    echo "  train:    $TRAIN_DIR" >&2
    echo "  validate: $VAL_DIR" >&2
    echo "Create symlinks or pass --no-data-check if this is intentional." >&2
    exit 3
  fi
fi

RUN_BATCH_ID="$(date +%Y_%m_%d_%H_%M_%S)"
BATCH_DIR="opencood/logs/ddsh_batch_runs/${DATASET}_${RUN_BATCH_ID}"
mkdir -p "$BATCH_DIR"
MANIFEST="$BATCH_DIR/manifest.csv"
echo "dataset,stage_label,yaml,run_dir,debug_log,train_log,infer_log,profile_log,status" \
  > "$MANIFEST"

yaml_name() {
  awk -F':' '/^name:/ {
    value=$2
    gsub(/^[ \t"]+/, "", value)
    gsub(/[ \t"]+$/, "", value)
    print value
    exit
  }' "$1"
}

latest_run_dir() {
  local name="$1"
  find opencood/logs -maxdepth 1 -type d -name "${name}_*" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

run_and_log() {
  local log_file="$1"
  shift
  echo "[DDSH batch] running: $*" | tee "$log_file"
  "$@" 2>&1 | tee -a "$log_file"
}

echo "DDSH batch run directory: $BATCH_DIR"
echo "Dataset: $DATASET"
echo "Stages: ${#YAMLS[@]}"

for YAML in "${YAMLS[@]}"; do
  if [[ ! -f "$YAML" ]]; then
    echo "YAML not found: $YAML" >&2
    exit 4
  fi

  NAME="$(yaml_name "$YAML")"
  STAGE_LABEL="$(basename "$YAML" .yaml)"
  DEBUG_LOG="$BATCH_DIR/${STAGE_LABEL}_debug.log"
  TRAIN_LOG="$BATCH_DIR/${STAGE_LABEL}_train.log"
  INFER_LOG="$BATCH_DIR/${STAGE_LABEL}_infer.log"
  PROFILE_LOG="$BATCH_DIR/${STAGE_LABEL}_profile.log"

  echo
  echo "============================================================"
  echo "Starting $STAGE_LABEL"
  echo "YAML: $YAML"
  echo "============================================================"

  if [[ "$RUN_DEBUG" == "1" ]]; then
    run_and_log "$DEBUG_LOG" \
      python3 opencood/tools/debug_ddsh_forward.py --hypes_yaml "$YAML"
  else
    echo "Skipped debug." | tee "$DEBUG_LOG"
  fi

  run_and_log "$TRAIN_LOG" \
    python3 opencood/tools/train.py --hypes_yaml "$YAML"

  RUN_DIR="$(latest_run_dir "$NAME")"
  if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
    echo "Could not locate run directory for YAML name=$NAME" | tee -a "$TRAIN_LOG"
    echo "${DATASET},${STAGE_LABEL},${YAML},,,,,failed_no_run_dir" >> "$MANIFEST"
    exit 5
  fi

  if [[ "$RUN_INFER" == "1" ]]; then
    run_and_log "$INFER_LOG" \
      python3 opencood/tools/inference.py \
        --model_dir "$RUN_DIR" \
        --fusion_method intermediate \
        --global_sort_detections
  else
    echo "Skipped inference." | tee "$INFER_LOG"
  fi

  if [[ "$RUN_PROFILE" == "1" ]]; then
    run_and_log "$PROFILE_LOG" \
      python3 opencood/tools/profile_ddsh_statistics.py \
        --model_dir "$RUN_DIR" \
        --split "$PROFILE_SPLIT" \
        --batch_size "$PROFILE_BATCH_SIZE" \
        --num_workers "$PROFILE_NUM_WORKERS" \
        --max_batches "$PROFILE_MAX_BATCHES"
  else
    echo "Skipped profile." | tee "$PROFILE_LOG"
  fi

  echo "${DATASET},${STAGE_LABEL},${YAML},${RUN_DIR},${DEBUG_LOG},${TRAIN_LOG},${INFER_LOG},${PROFILE_LOG},ok" \
    >> "$MANIFEST"
  echo "Finished $STAGE_LABEL"
  echo "Run dir: $RUN_DIR"
done

echo
echo "All requested DDSH stages finished."
echo "Manifest: $MANIFEST"
