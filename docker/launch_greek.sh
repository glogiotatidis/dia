#!/bin/bash
# Greek TTS Training Launch Script
#
# Usage:
#   ./docker/launch_greek.sh                    # Default settings
#   ./docker/launch_greek.sh --epochs 100       # Custom epochs
#   ./docker/launch_greek.sh --pretrained /path/to/model.pt  # Fine-tune
#
set -e

# Configuration
export HF_HOME=${HF_HOME:-/workspace/cache/huggingface}
export WANDB_PROJECT=${WANDB_PROJECT:-dia-greek-tts}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Paths
DATA_DIR=${DATA_DIR:-/workspace/data/el}
MANIFEST_PATH=${MANIFEST_PATH:-$DATA_DIR/manifests/train_manifest_el.json}
LANG_VOCAB=${LANG_VOCAB:-/workspace/configs/lang_vocab.json}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/checkpoints/greek}

# Training parameters (can be overridden via environment or args)
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-4}

# Parse command line arguments
PRETRAINED=""
USE_WANDB=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --pretrained)
            PRETRAINED="--pretrained $2"
            shift 2
            ;;
        --wandb)
            USE_WANDB="--wandb"
            shift
            ;;
        --manifest)
            MANIFEST_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "🇬🇷 Greek TTS Training Configuration"
echo "=============================================="
echo "Data directory:  $DATA_DIR"
echo "Manifest:        $MANIFEST_PATH"
echo "Language vocab:  $LANG_VOCAB"
echo "Output dir:      $OUTPUT_DIR"
echo "Epochs:          $EPOCHS"
echo "Batch size:      $BATCH_SIZE"
echo "Learning rate:   $LR"
echo "Pretrained:      ${PRETRAINED:-None}"
echo "W&B logging:     ${USE_WANDB:-disabled}"
echo "=============================================="

# Check if manifest exists
if [ ! -f "$MANIFEST_PATH" ]; then
    echo "❌ Manifest not found: $MANIFEST_PATH"
    echo ""
    echo "Run data preparation first:"
    echo "  python scripts/prepare_greek.py --source commonvoice --output_dir $DATA_DIR"
    exit 1
fi

# Check if lang vocab exists
if [ ! -f "$LANG_VOCAB" ]; then
    echo "❌ Language vocabulary not found: $LANG_VOCAB"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️  GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Run training
echo "🚀 Starting training..."
python3 scripts/train_greek.py \
    --manifest "$MANIFEST_PATH" \
    --lang_vocab "$LANG_VOCAB" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    $PRETRAINED \
    $USE_WANDB

echo ""
echo "✅ Training complete!"
echo "   Checkpoints saved to: $OUTPUT_DIR"
