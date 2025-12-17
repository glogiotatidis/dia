#!/bin/bash
# =============================================================================
# Greek TTS Training on RunPod
# =============================================================================
#
# USAGE:
#   bash train_runpod.sh              # Auto-detect GPU, quick test (5 epochs)
#   bash train_runpod.sh --full       # Auto-detect GPU, full training (50 epochs)
#   bash train_runpod.sh --a4000      # Force A4000/16GB settings
#   bash train_runpod.sh --a100       # Force A100/40GB settings
#   bash train_runpod.sh --a100 --full  # A100 full training
#   bash train_runpod.sh --lr 1e-6    # Use custom learning rate (for NaN issues)
#
# GPU PROFILES:
#   A4000/T4 (16GB): Frozen encoder, batch=1, max_audio=5s, lr=1e-5
#   A100 (40GB+):    Full model, batch=2, max_audio=10s, lr=5e-6
#
# NaN PREVENTION:
#   - GradScaler for mixed precision stability
#   - Gradient clipping (max_norm=1.0)
#   - Learning rate warmup (10% of steps or 500 steps)
#   - NaN detection and early stopping
#   - Automatic checkpoint skipping on bad epochs
#
# =============================================================================

# Don't exit on error during setup, only during training
# set -e

# Parse arguments
FULL_TRAINING=false
GPU_PROFILE="auto"
CUSTOM_LR=""

for arg in "$@"; do
    case $arg in
        --full)
            FULL_TRAINING=true
            ;;
        --a4000|--t4|--16gb)
            GPU_PROFILE="low"
            ;;
        --a100|--4090|--40gb|--high)
            GPU_PROFILE="high"
            ;;
        --lr)
            # Next argument will be the learning rate
            NEXT_IS_LR=true
            ;;
        *)
            if [ "$NEXT_IS_LR" = true ]; then
                CUSTOM_LR="$arg"
                NEXT_IS_LR=false
            fi
            ;;
    esac
done

# Auto-detect GPU if not specified
if [ "$GPU_PROFILE" == "auto" ]; then
    GPU_MEM=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory // (1024**3))" 2>/dev/null || echo "0")
    if [ "$GPU_MEM" -ge 30 ]; then
        GPU_PROFILE="high"
        echo "🔍 Detected ${GPU_MEM}GB GPU → Using A100/high-memory settings"
    else
        GPU_PROFILE="low"
        echo "🔍 Detected ${GPU_MEM}GB GPU → Using A4000/low-memory settings"
    fi
fi

# Set GPU-specific parameters
if [ "$GPU_PROFILE" == "high" ]; then
    # A100 / RTX 4090 / 40GB+ settings
    BATCH_SIZE=2
    GRAD_ACCUM=4
    MAX_AUDIO_LEN=10.0
    FREEZE_ENCODER=""  # Don't freeze, train full model
    GPU_NAME="A100/40GB+"
    # Slightly higher LR for full model training
    DEFAULT_LR="5e-6"
else
    # A4000 / T4 / 16GB settings
    BATCH_SIZE=1
    GRAD_ACCUM=4
    MAX_AUDIO_LEN=5.0
    FREEZE_ENCODER="--freeze_encoder"
    GPU_NAME="A4000/16GB"
    # Lower LR for stability with frozen encoder
    DEFAULT_LR="1e-5"
fi

# Use custom LR if provided, otherwise use default
if [ -n "$CUSTOM_LR" ]; then
    LEARNING_RATE="$CUSTOM_LR"
    echo "📝 Using custom learning rate: $LEARNING_RATE"
else
    LEARNING_RATE="$DEFAULT_LR"
fi

# Set epochs based on training mode
if [ "$FULL_TRAINING" == "true" ]; then
    EPOCHS=50
    MODE_NAME="FULL"
else
    EPOCHS=5
    MODE_NAME="Quick Test"
fi

echo "=============================================="
echo "🚀 Greek TTS Training (RunPod)"
echo "=============================================="
echo "   GPU Profile: $GPU_NAME"
echo "   Mode: $MODE_NAME ($EPOCHS epochs)"
echo "   Batch: $BATCH_SIZE x $GRAD_ACCUM accum = $(($BATCH_SIZE * $GRAD_ACCUM)) effective"
echo "   Max audio: ${MAX_AUDIO_LEN}s"
[ -n "$FREEZE_ENCODER" ] && echo "   Encoder: FROZEN (decoder only)"
[ -z "$FREEZE_ENCODER" ] && echo "   Encoder: TRAINABLE (full model)"
echo "=============================================="
echo ""

# Install system dependencies
echo "📦 Installing system packages..."
apt-get update > /dev/null 2>&1 || echo "apt-get update failed (may need sudo)"
apt-get install -y espeak-ng libsndfile1 ffmpeg git > /dev/null 2>&1 || echo "Some packages may already be installed"
echo "✅ System packages done"

# Install Python dependencies
echo "📦 Checking Python packages..."

# Check if PyTorch needs reinstall
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
TORCH_CUDA=$(python -c "import torch; print('cu118' if 'cu118' in torch.__file__ or torch.version.cuda == '11.8' else 'other')" 2>/dev/null || echo "none")

if [[ "$TORCH_VERSION" != "2.1.0"* ]] || [ "$TORCH_CUDA" != "cu118" ]; then
    echo "📦 Installing PyTorch 2.1.0 with CUDA 11.8 (current: $TORCH_VERSION, cuda: $TORCH_CUDA)..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install -q torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
else
    echo "✅ PyTorch $TORCH_VERSION already installed with CUDA 11.8"
fi

# Check transformers version
TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")
TRANSFORMERS_OK=$(python -c "
import transformers
v = tuple(map(int, transformers.__version__.split('.')[:2]))
print('ok' if (4, 35) <= v < (4, 40) else 'no')
" 2>/dev/null || echo "no")

if [ "$TRANSFORMERS_OK" != "ok" ]; then
    echo "📦 Installing compatible transformers (current: $TRANSFORMERS_VERSION)..."
    pip uninstall -y transformers 2>/dev/null || true
    pip install -q 'transformers>=4.35.0,<4.40.0'
else
    echo "✅ Transformers $TRANSFORMERS_VERSION already compatible"
fi

# Install other dependencies (pip will skip if already satisfied)
pip install -q huggingface_hub speechbrain tqdm einops soundfile librosa pydantic 'datasets>=2.14.0,<3.0.0'
pip install -q descript-audio-codec
echo "✅ Python packages ready"

# Clone repo
REPO_DIR="/workspace/dia"
if [ ! -d "$REPO_DIR" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/glogiotatidis/dia.git $REPO_DIR
fi
cd $REPO_DIR
git pull  # Get latest changes
echo "📂 Working directory: $(pwd)"

# Setup directories
DATA_DIR="/workspace/data/el"
CHECKPOINT_DIR="/workspace/checkpoints/greek"
mkdir -p $CHECKPOINT_DIR $DATA_DIR

# Extract Fleurs data (greek_data.tar.gz)
if [ -f "/workspace/greek_data.tar.gz" ]; then
    echo "📦 Extracting Fleurs data..."
    tar -xzf /workspace/greek_data.tar.gz -C /workspace/data
    
    # Fix audio paths in manifests (replace original path with actual path)
    echo "🔧 Fixing manifest paths..."
    for manifest in $DATA_DIR/manifests/*.json; do
        if [ -f "$manifest" ]; then
            # Replace any path prefix with the actual data directory
            sed -i 's|"audio": "[^"]*/fleurs/audio/|"audio": "'$DATA_DIR'/fleurs/audio/|g' "$manifest"
            sed -i 's|"audio": "[^"]*/commonvoice/audio/|"audio": "'$DATA_DIR'/commonvoice/audio/|g' "$manifest"
        fi
    done
    echo "✅ Fleurs data extracted and paths fixed"
fi

# Extract CommonVoice data if available (for full training)
if [ "$FULL_TRAINING" == "true" ] && [ -f "/workspace/mvc-scripted-el-v24.0.tar.gz" ]; then
    echo "📦 Extracting CommonVoice data..."
    mkdir -p $DATA_DIR/commonvoice
    tar -xzf /workspace/mvc-scripted-el-v24.0.tar.gz -C $DATA_DIR/commonvoice
    echo "✅ CommonVoice data extracted"
fi

# Check for manifest
if [ ! -f "$DATA_DIR/manifests/train_manifest_el.json" ]; then
    echo "❌ Manifest not found at $DATA_DIR/manifests/train_manifest_el.json"
    echo ""
    echo "Please upload your data first:"
    echo "  1. On your local machine:"
    echo "     python scripts/download_greek_datasets.py --output_dir data/el"
    echo "     tar -czvf greek_data.tar.gz -C data el"
    echo ""
    echo "  2. Upload to RunPod:"
    echo "     runpodctl send greek_data.tar.gz"
    echo ""
    echo "  3. Then run this script again"
    exit 1
fi

N_SAMPLES=$(python -c "import json; print(len(json.load(open('$DATA_DIR/manifests/train_manifest_el.json'))))")
echo "✅ Found $N_SAMPLES training samples"

# Estimate training time
if [ "$GPU_PROFILE" == "high" ]; then
    if [ "$FULL_TRAINING" == "true" ]; then
        EST_TIME="4-6 hours"
    else
        EST_TIME="20-30 minutes"
    fi
else
    if [ "$FULL_TRAINING" == "true" ]; then
        EST_TIME="8-12 hours"
    else
        EST_TIME="1-2 hours"
    fi
fi

echo ""
echo "🏋️ Training Configuration:"
echo "   Epochs: $EPOCHS"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Estimated time: $EST_TIME"
echo ""
echo "   Monitor with: tail -f $CHECKPOINT_DIR/training.log"
echo ""

# Clear GPU memory before starting
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Set environment variables for stability
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

echo ""
echo "🚀 Starting training..."
echo "   (NaN detection enabled - training will stop if loss diverges)"
echo ""

# Start training (exit on failure)
set -e
python $REPO_DIR/scripts/train_greek.py \
    --manifest $DATA_DIR/manifests/train_manifest_el.json \
    --output_dir $CHECKPOINT_DIR \
    --from_hf \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --lr $LEARNING_RATE \
    --max_audio_len $MAX_AUDIO_LEN \
    $FREEZE_ENCODER \
    2>&1 | tee $CHECKPOINT_DIR/training.log

# Check if training succeeded (look for best checkpoint)
if [ -f "$CHECKPOINT_DIR/greek_best.pt" ]; then
    echo ""
    echo "=============================================="
    echo "✅ Training Complete!"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "⚠️  Training may have failed - no best checkpoint found"
    echo "=============================================="
    echo "Check the log for errors: $CHECKPOINT_DIR/training.log"
    echo ""
    echo "Common issues:"
    echo "  - NaN loss: Try lower learning rate (--lr 1e-6)"
    echo "  - OOM: Reduce batch size or max_audio_len"
    echo "  - Bad audio files: Check dataset for corrupt files"
fi

echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "To download your model:"
echo "  runpodctl receive"
echo "  # or: scp user@pod:/workspace/checkpoints/greek/greek_best.pt ."
echo ""
echo "To test your model:"
echo "  python scripts/infer_greek.py --checkpoint $CHECKPOINT_DIR/greek_best.pt --text 'Γεια σου κόσμε'"
echo ""
if [ "$FULL_TRAINING" == "false" ]; then
    echo "This was a quick test. For full training, run:"
    echo "  bash scripts/train_runpod.sh --full"
    echo ""
fi
