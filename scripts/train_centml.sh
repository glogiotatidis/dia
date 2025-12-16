#!/bin/bash
# =============================================================================
# Greek TTS Training on CentML
# =============================================================================
#
# SETUP INSTRUCTIONS:
#
# 1. Create account at https://centml.ai
# 2. Add credits or connect payment method
# 3. Create a new Workload:
#    - Image: PyTorch 2.1 + CUDA 12.1
#    - GPU: A100 40GB (recommended) or A10 (budget option)
#    - Storage: 100GB
#    - Region: Any (choose cheapest)
#
# 4. SSH into your instance or use the web terminal
#
# 5. Run this script:
#    wget https://raw.githubusercontent.com/YOUR_REPO/scripts/train_centml.sh
#    bash train_centml.sh
#
# COST ESTIMATE:
#   - A100 @ $1.30/hr × 10 hours = ~$13 for full training
#   - Spot A100 @ $0.60/hr × 10 hours = ~$6 (if available)
#
# =============================================================================

set -e

echo "=============================================="
echo "🚀 Greek TTS Training Setup (CentML)"
echo "=============================================="

# Print GPU info
echo ""
echo "🖥️  GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq espeak-ng git wget > /dev/null 2>&1
echo "✅ espeak-ng installed"

# Verify espeak Greek support
echo -n "🇬🇷 Testing Greek phonemization: "
echo "Γεια σας" | espeak-ng -v el --ipa -q
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers datasets huggingface_hub speechbrain phonemizer tqdm wandb einops

# Clone repository
WORK_DIR="/workspace"
mkdir -p $WORK_DIR
cd $WORK_DIR

if [ ! -d "dia" ]; then
    echo "📥 Cloning DIA repository..."
    git clone https://github.com/nari-labs/dia.git
fi
cd dia

# Create data directory
DATA_DIR="$WORK_DIR/data/el"
CHECKPOINT_DIR="$WORK_DIR/checkpoints/greek"
mkdir -p $DATA_DIR $CHECKPOINT_DIR

# HuggingFace login
echo ""
echo "🔐 HuggingFace Login Required"
echo "   Get your token from: https://huggingface.co/settings/tokens"
echo "   (Need 'read' access, and accept Common Voice license)"
echo ""
huggingface-cli login

# Download datasets
echo ""
echo "📥 Downloading Greek speech datasets..."
echo "   This may take 10-30 minutes depending on connection..."
echo ""

python scripts/download_greek_datasets.py \
    --datasets commonvoice voxpopuli \
    --output_dir $DATA_DIR \
    --max_samples 30000

# Verify data
echo ""
echo "📊 Dataset Summary:"
python -c "
import json
with open('$DATA_DIR/manifests/train_manifest_el.json') as f:
    data = json.load(f)
total_hours = sum(s.get('duration', 0) for s in data) / 3600
print(f'   Training samples: {len(data)}')
print(f'   Total audio: {total_hours:.1f} hours')
"

# Calculate optimal batch size based on GPU
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -ge 70000 ]; then
    BATCH_SIZE=64
    echo "   Using batch_size=64 (A100 80GB/H100 detected)"
elif [ "$GPU_MEM" -ge 35000 ]; then
    BATCH_SIZE=32
    echo "   Using batch_size=32 (A100 40GB detected)"
elif [ "$GPU_MEM" -ge 20000 ]; then
    BATCH_SIZE=16
    echo "   Using batch_size=16 (A10/RTX detected)"
else
    BATCH_SIZE=8
    echo "   Using batch_size=8 (Limited VRAM)"
fi

# Optional: Setup W&B
echo ""
read -p "Enable Weights & Biases logging? (y/n): " USE_WANDB
WANDB_FLAG=""
if [ "$USE_WANDB" = "y" ]; then
    wandb login
    WANDB_FLAG="--wandb"
fi

# Start training
echo ""
echo "=============================================="
echo "🏋️  Starting Training"
echo "=============================================="
echo "   Batch size: $BATCH_SIZE"
echo "   Epochs: 50"
echo "   Estimated time: 8-12 hours on A100"
echo ""
echo "   Monitor progress: tail -f $WORK_DIR/training.log"
echo "   To resume if interrupted: add --pretrained flag"
echo "=============================================="
echo ""

# Run training with logging
python scripts/train_greek.py \
    --manifest $DATA_DIR/manifests/train_manifest_el.json \
    --lang_vocab configs/lang_vocab.json \
    --output_dir $CHECKPOINT_DIR \
    --epochs 50 \
    --batch_size $BATCH_SIZE \
    --lr 1e-4 \
    $WANDB_FLAG 2>&1 | tee $WORK_DIR/training.log

# Training complete
echo ""
echo "=============================================="
echo "✅ Training Complete!"
echo "=============================================="
echo ""
echo "📁 Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "Best model: $CHECKPOINT_DIR/greek_best.pt"
echo ""
echo "To test inference:"
echo "  python scripts/infer_greek.py \\"
echo "      --model_path $CHECKPOINT_DIR/greek_best.pt \\"
echo "      --text 'Γεια σας!' \\"
echo "      --output_dir samples/"
echo ""
echo "To download model locally, use SCP or CentML's file browser."
echo ""
