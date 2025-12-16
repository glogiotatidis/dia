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
# 4. Connect via SSH or Web Terminal
#
# 5. Run this script:
#    curl -O https://raw.githubusercontent.com/YOUR_REPO/scripts/train_runpod.sh
#    bash train_runpod.sh
#
# =============================================================================

set -e

echo "=============================================="
echo "🚀 Greek TTS Training Setup (RunPod)"
echo "=============================================="

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers datasets huggingface_hub speechbrain torchaudio phonemizer tqdm

# Install espeak-ng
apt-get update && apt-get install -y espeak-ng > /dev/null 2>&1
echo "✅ espeak-ng installed"

# Clone repo
if [ ! -d "dia-multilingual" ]; then
    echo "📥 Cloning repository..."
    git clone https://github.com/nari-labs/dia.git
    cd dia/dia-multilingual
else
    cd dia-multilingual
fi

# Login to HuggingFace (for Common Voice)
echo ""
echo "🔐 HuggingFace Login"
echo "   Get token from: https://huggingface.co/settings/tokens"
huggingface-cli login

# Download data
echo ""
echo "📥 Downloading Greek datasets..."
python scripts/download_greek_datasets.py \
    --datasets commonvoice fleurs \
    --output_dir /workspace/data/el \
    --max_samples 20000  # Adjust based on time/budget

# Check data
N_SAMPLES=$(cat /workspace/data/el/manifests/train_manifest_el.json | python -c "import json,sys; print(len(json.load(sys.stdin)))")
echo "✅ Downloaded $N_SAMPLES training samples"

# Start training
echo ""
echo "🏋️ Starting training..."
echo "   This will take ~8-12 hours on A100"
echo "   Monitor with: tail -f training.log"
echo ""

python scripts/train_greek.py \
    --manifest /workspace/data/el/manifests/train_manifest_el.json \
    --lang_vocab configs/lang_vocab.json \
    --output_dir /workspace/checkpoints/greek \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --wandb 2>&1 | tee training.log

echo ""
echo "=============================================="
echo "✅ Training Complete!"
echo "=============================================="
echo "Checkpoints saved to: /workspace/checkpoints/greek"
echo ""
echo "To download your model:"
echo "  runpodctl send /workspace/checkpoints/greek/greek_best.pt"
echo ""
