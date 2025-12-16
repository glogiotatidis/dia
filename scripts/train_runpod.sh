#!/bin/bash
# =============================================================================
# Greek TTS Training on RunPod
# =============================================================================
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
#    - Upload greek_data.tar.gz to /workspace/
#    - Or use runpodctl: runpodctl send greek_data.tar.gz
#
# 5. Connect via SSH or Web Terminal
#
# 6. Run this script:
#    bash train_runpod.sh
#
# =============================================================================

set -e

echo "=============================================="
echo "🚀 Greek TTS Training Setup (RunPod)"
echo "=============================================="

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers huggingface_hub speechbrain torchaudio tqdm einops

# Install espeak-ng
apt-get update && apt-get install -y espeak-ng > /dev/null 2>&1
echo "✅ espeak-ng installed"

# Clone repo
if [ ! -d "dia" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/nari-labs/dia.git
fi
cd dia

# Setup data directory
DATA_DIR="/workspace/data/el"
CHECKPOINT_DIR="/workspace/checkpoints/greek"
mkdir -p $CHECKPOINT_DIR

# Check for pre-uploaded data
if [ -f "/workspace/greek_data.tar.gz" ]; then
    echo "📦 Extracting pre-uploaded data..."
    mkdir -p /workspace/data
    tar -xzf /workspace/greek_data.tar.gz -C /workspace/data
    echo "✅ Data extracted to $DATA_DIR"
elif [ -d "$DATA_DIR/manifests" ]; then
    echo "✅ Data already exists at $DATA_DIR"
else
    echo "❌ No data found!"
    echo ""
    echo "Please upload your data first:"
    echo "  1. On your local machine, prepare data:"
    echo "     python scripts/download_greek_datasets.py --output_dir data/el"
    echo "     tar -czvf greek_data.tar.gz -C data el"
    echo ""
    echo "  2. Upload to RunPod:"
    echo "     runpodctl send greek_data.tar.gz"
    echo ""
    echo "  3. Then run this script again"
    exit 1
fi

# Verify data
if [ ! -f "$DATA_DIR/manifests/train_manifest_el.json" ]; then
    echo "❌ Manifest not found at $DATA_DIR/manifests/train_manifest_el.json"
    exit 1
fi

N_SAMPLES=$(python -c "import json; print(len(json.load(open('$DATA_DIR/manifests/train_manifest_el.json'))))")
echo "✅ Found $N_SAMPLES training samples"

# Start training
echo ""
echo "🏋️ Starting training..."
echo "   This will take ~8-12 hours on A100"
echo "   Monitor with: tail -f training.log"
echo ""

python scripts/train_greek.py \
    --manifest $DATA_DIR/manifests/train_manifest_el.json \
    --lang_vocab configs/lang_vocab.json \
    --output_dir $CHECKPOINT_DIR \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 2>&1 | tee training.log

echo ""
echo "=============================================="
echo "✅ Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "To download your model:"
echo "  runpodctl send $CHECKPOINT_DIR/greek_best.pt"
echo ""
