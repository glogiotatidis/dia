#!/usr/bin/env python3
"""
Greek TTS Inference Script

Generate Greek speech from text using trained DIA model.

Usage:
    # Basic inference
    python scripts/infer_greek.py \
        --model_path checkpoints/greek/greek_best.pt \
        --text "Γεια σας, καλώς ήρθατε" \
        --output_dir samples/

    # With reference audio (voice cloning)
    python scripts/infer_greek.py \
        --model_path checkpoints/greek/greek_best.pt \
        --text "Πώς είστε σήμερα;" \
        --reference_wav samples/greek_speaker.wav \
        --output_dir samples/
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import torch
import torchaudio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dia.model import DiaModel
from tools.speaker_encoder import SpeakerEncoder


def phonemize_greek(text: str) -> str:
    """Convert Greek text to IPA phonemes using espeak-ng."""
    text = re.sub(r'[^\w\s\u0370-\u03FF\u1F00-\u1FFF.,;!?]', '', text)
    text = text.strip()
    
    if not text:
        return ""
    
    try:
        result = subprocess.run(
            ["espeak-ng", "-v", "el", "--ipa", "-q", text],
            capture_output=True,
            text=True,
            timeout=10
        )
        phonemes = result.stdout.strip()
        phonemes = re.sub(r'\s+', ' ', phonemes)
        return phonemes
    except Exception as e:
        print(f"Warning: Phonemization failed: {e}")
        return ""


def build_phoneme_vocab(phonemes: str) -> dict:
    """Build a simple phoneme vocabulary from input."""
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, char in enumerate(sorted(set(phonemes))):
        vocab[char] = i + 10
    return vocab


def main():
    parser = argparse.ArgumentParser(description="Greek TTS Inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--lang_vocab", type=str, default="configs/lang_vocab.json",
                        help="Path to language vocabulary")
    parser.add_argument("--text", type=str, required=True,
                        help="Greek text to synthesize")
    parser.add_argument("--output_dir", type=str, default="samples",
                        help="Output directory for generated audio")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename (without extension)")
    parser.add_argument("--reference_wav", type=str, default=None,
                        help="Reference audio for voice cloning")
    parser.add_argument("--sample_rate", type=int, default=22050,
                        help="Output sample rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps)")
    
    args = parser.parse_args()
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    
    # Load language vocabulary
    with open(args.lang_vocab) as f:
        lang_vocab = json.load(f)
    
    lang_token = "<el>"
    lang_token_id = lang_vocab.get(lang_token, 0)
    print(f"🇬🇷 Language: Greek (token_id={lang_token_id})")
    
    # Phonemize input text
    print(f"\n📝 Input text: {args.text}")
    phonemes = phonemize_greek(args.text)
    
    if not phonemes:
        print("❌ Failed to phonemize text")
        return
    
    print(f"🔤 Phonemes: {phonemes}")
    
    # Build phoneme IDs
    phoneme_vocab = build_phoneme_vocab(phonemes)
    phoneme_ids = [phoneme_vocab.get(c, 1) for c in phonemes]
    input_ids = torch.tensor([lang_token_id] + phoneme_ids).unsqueeze(0).to(device)
    
    # Load model
    print(f"\n📥 Loading model from {args.model_path}")
    
    # Create a simple config for model initialization
    config = {
        "model": {
            "encoder_vocab_size": 512,
            "decoder": {"d_model": 512},
            "tgt_vocab_size": 1028,
            "input_dim": 80,
            "diffusion_steps": 8,
        }
    }
    
    model = DiaModel(config["model"])
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Process reference audio if provided
    ref_audio = None
    spk_embed = None
    
    if args.reference_wav:
        print(f"🎤 Loading reference audio: {args.reference_wav}")
        wav, sr = torchaudio.load(args.reference_wav)
        if sr != args.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, args.sample_rate)
        ref_audio = wav.squeeze(0).unsqueeze(0).to(device)
        
        # Get speaker embedding
        spk_encoder = SpeakerEncoder(device=device)
        spk_embed = spk_encoder.encode(args.reference_wav).unsqueeze(0).to(device)
    
    # Generate audio
    print("\n🎵 Generating audio...")
    
    with torch.no_grad():
        # Build batch
        batch = {
            "input_ids": input_ids,
            "lang_token_ids": torch.tensor([lang_token_id]).to(device),
        }
        
        if spk_embed is not None:
            batch["spk_embed"] = spk_embed
        
        if ref_audio is not None:
            batch["ref_audio"] = ref_audio
        
        # Inference (model.infer method should be implemented)
        if hasattr(model, 'infer'):
            audio = model.infer(batch)
        else:
            # Fallback: use forward pass and extract audio
            # This is a placeholder - actual inference depends on model architecture
            print("⚠️  Using fallback inference mode")
            output = model.decoder_forward(
                ref_audio if ref_audio is not None else torch.randn(1, 100, 80).to(device),
                spk_embed=spk_embed,
                encoder_out=model.encoder(input_ids, batch["lang_token_ids"])
            )
            audio = output.squeeze(0)
    
    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output_name:
        output_name = args.output_name
    else:
        # Generate name from text
        safe_text = re.sub(r'[^\w\s]', '', args.text[:30]).replace(' ', '_')
        output_name = f"greek_{safe_text}"
    
    output_path = output_dir / f"{output_name}.wav"
    
    # Ensure audio is the right shape for saving
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() > 2:
        audio = audio.squeeze()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
    
    torchaudio.save(str(output_path), audio.cpu(), args.sample_rate)
    
    print(f"\n✅ Audio saved to: {output_path}")
    print(f"   Duration: {audio.shape[-1] / args.sample_rate:.2f}s")


if __name__ == "__main__":
    main()
