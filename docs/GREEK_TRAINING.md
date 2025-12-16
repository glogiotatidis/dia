# 🇬🇷 Greek (el) Language Training Guide

This guide covers everything you need to train the DIA TTS model for Greek language.

## Overview

Greek (`el`) is already supported in the language vocabulary with **token_id=12**. The training pipeline uses:
- **espeak-ng** for phonemization (Greek IPA)
- **SpeechBrain Conformer** for speaker embeddings
- **Diffusion-based decoder** for high-quality audio generation

---

## Prerequisites

### 1. System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended (RTX 3090, A100, etc.)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for datasets and checkpoints
- **OS**: Linux (Ubuntu 22.04 recommended) or macOS

### 2. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/nari-labs/dia.git
cd dia

# Install Python dependencies
pip install -r requirements.txt
# Or using uv:
uv sync

# Install espeak-ng (required for phonemization)
# Ubuntu/Debian:
sudo apt-get install espeak-ng espeak-ng-data

# macOS:
brew install espeak-ng

# Verify Greek support
espeak-ng --voices=el
```

### 3. Verify espeak-ng Greek Support

```bash
# Test Greek phonemization
echo "Γεια σας" | espeak-ng -v el --ipa -q
# Expected output: ʝa sas
```

---

## Data Preparation

### Option 1: Mozilla Common Voice (Recommended)

The easiest way to get Greek training data:

```bash
# Download and prepare Common Voice Greek data
python scripts/prepare_greek.py \
    --source commonvoice \
    --output_dir data/el \
    --max_samples 50000  # Limit samples for testing
```

**Note**: You need to:
1. Create a [HuggingFace account](https://huggingface.co/join)
2. Accept the [Common Voice license](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
3. Login: `huggingface-cli login`

### Option 2: Local Data

If you have your own Greek audio dataset:

1. **Prepare your data** in this structure:
   ```
   raw_data/
   ├── audio/
   │   ├── sample1.wav
   │   ├── sample2.wav
   │   └── ...
   └── transcripts.tsv  # or transcripts.json
   ```

2. **Transcript format** (TSV):
   ```
   sample1.wav	Γεια σας, καλώς ήρθατε
   sample2.wav	Πώς είστε σήμερα;
   ```

3. **Clean and prepare**:
   ```bash
   # First, clean the audio
   python tools/greek_audio_cleaner.py \
       --input_dir raw_data/audio \
       --output_dir data/el/audio \
       --sample_rate 22050

   # Then prepare the manifest
   python scripts/prepare_greek.py \
       --source local \
       --audio_dir data/el/audio \
       --transcript_file raw_data/transcripts.tsv \
       --output_dir data/el
   ```

### Audio Requirements

For best results, your audio should be:
- **Sample rate**: 22050 Hz (will be resampled if different)
- **Format**: WAV, MP3, or FLAC
- **Duration**: 1-15 seconds per utterance
- **Quality**: Clean speech, minimal background noise
- **Speakers**: Multiple speakers recommended for better generalization

---

## Training

### Basic Training

```bash
# Train from scratch on Greek data
python scripts/train_greek.py \
    --manifest data/el/manifests/train_manifest_el.json \
    --lang_vocab configs/lang_vocab.json \
    --output_dir checkpoints/greek \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4
```

### Fine-tuning from Pretrained

If you have a pretrained multilingual checkpoint:

```bash
python scripts/train_greek.py \
    --manifest data/el/manifests/train_manifest_el.json \
    --pretrained checkpoints/multilang_base.pt \
    --output_dir checkpoints/greek_finetuned \
    --epochs 20 \
    --lr 5e-5  # Lower LR for fine-tuning
```

### Docker Training (GPU)

For reproducible training with Docker:

```bash
# Build the container
docker build -t dia-greek -f docker/Dockerfile .

# Run training
docker run --gpus all -v $(pwd)/data:/workspace/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    dia-greek bash docker/launch_greek.sh
```

### Training with Weights & Biases

Enable experiment tracking:

```bash
# Set your W&B API key
export WANDB_API_KEY=your_key_here

python scripts/train_greek.py \
    --manifest data/el/manifests/train_manifest_el.json \
    --output_dir checkpoints/greek \
    --wandb
```

---

## Inference

### Generate Greek Speech

```bash
python scripts/infer_greek.py \
    --model_path checkpoints/greek/greek_best.pt \
    --text "Γεια σας, καλώς ήρθατε στην Ελλάδα!" \
    --output_dir samples/
```

### Voice Cloning

Use a reference audio to clone the speaker's voice:

```bash
python scripts/infer_greek.py \
    --model_path checkpoints/greek/greek_best.pt \
    --text "Η φωνή μου θα ακούγεται σαν τον ομιλητή" \
    --reference_wav samples/greek_speaker.wav \
    --output_dir samples/
```

---

## Training Tips

### Data Quality

1. **Clean audio is crucial**: Use `greek_audio_cleaner.py` to preprocess
2. **Balanced dataset**: Include various speakers, ages, and accents
3. **Transcript accuracy**: Phonemization depends on correct text

### Hyperparameters

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| Batch size | 16-32 | Increase if GPU memory allows |
| Learning rate | 1e-4 (scratch), 5e-5 (fine-tune) | Use warmup |
| Epochs | 50-100 | Monitor validation loss |
| Audio length | 1-15 seconds | Filter outliers |

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size
--batch_size 8

# Or use gradient accumulation (if supported)
--gradient_accumulation_steps 2
```

**2. Poor phonemization**
- Ensure espeak-ng is properly installed with Greek data
- Check: `espeak-ng --voices | grep el`

**3. Noisy outputs**
- Increase training epochs
- Use cleaner training data
- Add more speaker diversity

---

## File Structure

After preparation, your directory should look like:

```
dia/
├── configs/
│   ├── lang_vocab.json          # Language tokens (el = 12)
│   └── train_manifest_el.json   # Template manifest
├── data/
│   └── el/
│       ├── audio/               # Processed audio files
│       └── manifests/
│           ├── train_manifest_el.json
│           ├── val_manifest_el.json
│           ├── test_manifest_el.json
│           └── phoneme_vocab_el.json
├── checkpoints/
│   └── greek/
│       ├── greek_best.pt
│       ├── greek_latest.pt
│       └── greek_epoch*.pt
└── samples/                     # Generated audio
```

---

## Greek-Specific Notes

### Greek Phonemes

espeak-ng produces IPA phonemes for Greek. Common phonemes include:
- Vowels: `a, e, i, o, u`
- Consonants: `p, t, k, b, d, g, m, n, l, r, s, z, f, v, θ, ð, x, ɣ`
- Special: `ʝ` (γεια), `c` (κι)

### Text Normalization

The preparation script handles:
- Greek Unicode characters (U+0370-U+03FF)
- Polytonic Greek (U+1F00-U+1FFF)
- Common punctuation

### Speaker Diversity

Greek has distinct regional accents:
- Standard (Athenian)
- Northern Greek
- Cretan
- Cypriot

For best results, include speakers from various regions.

---

## Next Steps

1. **Prepare your data** using the scripts provided
2. **Start training** with default parameters
3. **Monitor** using W&B or TensorBoard
4. **Evaluate** on held-out test set
5. **Fine-tune** hyperparameters as needed

For questions or issues, open a GitHub issue on the DIA repository.

---

## Quick Start Commands

```bash
# Full pipeline from scratch
cd dia

# 1. Prepare data (downloads ~5GB)
python scripts/prepare_greek.py --source commonvoice --output_dir data/el

# 2. Train model
python scripts/train_greek.py \
    --manifest data/el/manifests/train_manifest_el.json \
    --output_dir checkpoints/greek \
    --epochs 50

# 3. Generate speech
python scripts/infer_greek.py \
    --model_path checkpoints/greek/greek_best.pt \
    --text "Γεια σας!" \
    --output_dir samples/
```

Happy training! 🚀🇬🇷
