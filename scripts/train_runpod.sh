#!/bin/bash
# =============================================================================
# Greek TTS Training on RunPod
# =============================================================================
#
# USAGE:
#   bash train_runpod.sh          # Quick test run (5 epochs, small batch)
#   bash train_runpod.sh --full   # Full training (50 epochs, uses CommonVoice if available)
#
# SETUP INSTRUCTIONS:
#
# 1. Create account at https://runpod.io
# 2. Add credits ($20-50 recommended for testing)
# 3. Deploy a GPU Pod:
#    - Template: RunPod Pytorch 2.1
#    - GPU: RTX A4000 ($0.20/hr) for testing, A100 ($1.50/hr) for full training
#    - Disk: 50GB minimum
#
# 4. Upload your pre-prepared data:
#    - Quick test: Upload greek_data.tar.gz (Fleurs, ~1GB)
#    - Full train: Also upload mvc-scripted-el-v24.0.tar.gz (CommonVoice)
#
# 5. Connect via SSH or Web Terminal and run this script
#
# =============================================================================

set -e

# Parse arguments
FULL_TRAINING=false
if [ "$1" == "--full" ]; then
    FULL_TRAINING=true
fi

if [ "$FULL_TRAINING" == "true" ]; then
    echo "=============================================="
    echo "🚀 Greek TTS FULL Training (RunPod)"
    echo "=============================================="
    EPOCHS=50
    BATCH_SIZE=4  # Reduced for DAC + full model memory
    MAX_SAMPLES=""
else
    echo "=============================================="
    echo "🧪 Greek TTS Quick Test (RunPod)"
    echo "   Run with --full for complete training"
    echo "=============================================="
    EPOCHS=5
    BATCH_SIZE=2  # Small batch for testing
    MAX_SAMPLES="--max_samples 500"
fi

# Install system dependencies
echo "📦 Installing system packages..."
apt-get update && apt-get install -y \
    espeak-ng \
    libsndfile1 \
    ffmpeg \
    git \
    > /dev/null 2>&1
echo "✅ System packages installed"

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
    
    # TODO: Process CommonVoice and merge manifests if needed
    # For now, we use Fleurs manifest
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

# Training configuration
echo ""
echo "🏋️ Training Configuration:"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
if [ "$FULL_TRAINING" == "true" ]; then
    echo "   Mode: Full training"
    echo "   Estimated time: ~8-12 hours on A100"
else
    echo "   Mode: Quick test"
    echo "   Estimated time: ~30-60 minutes"
fi
echo ""
echo "   Monitor with: tail -f training.log"
echo ""

# Start training
python $REPO_DIR/scripts/train_greek.py \
    --manifest $DATA_DIR/manifests/train_manifest_el.json \
    --lang_vocab $REPO_DIR/configs/lang_vocab.json \
    --output_dir $CHECKPOINT_DIR \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 1e-4 2>&1 | tee $CHECKPOINT_DIR/training.log

echo ""
echo "=============================================="
echo "✅ Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "To download your model:"
echo "  runpodctl send $CHECKPOINT_DIR/greek_best.pt"
echo ""
if [ "$FULL_TRAINING" == "false" ]; then
    echo "This was a quick test. For full training, run:"
    echo "  bash train_runpod.sh --full"
    echo ""
fi
