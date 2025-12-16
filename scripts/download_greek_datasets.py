#!/usr/bin/env python3
"""
Download and prepare Greek speech datasets for TTS training.

Uses Google Fleurs (el_gr) - freely available, ~3000 samples

Usage:
    python scripts/download_greek_datasets.py --output_dir data/el
    python scripts/download_greek_datasets.py --output_dir data/el --max_samples 1000
"""

import argparse
import json
import subprocess
import re
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import soundfile as sf
from tqdm import tqdm


def save_audio(path: str, wav: torch.Tensor, sr: int):
    """Save audio using soundfile (more reliable than torchaudio.save)."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    wav_np = wav.squeeze(0).numpy() if wav.shape[0] == 1 else wav.T.numpy()
    sf.write(path, wav_np, sr)


def _load_dataset_compat(dataset_name: str, lang: str, split: str = "train"):
    """Load dataset with compatibility for different datasets library versions."""
    from datasets import load_dataset
    import datasets
    
    major_version = int(datasets.__version__.split('.')[0])
    
    if major_version >= 3:
        return load_dataset(dataset_name, lang, split=split)
    else:
        return load_dataset(dataset_name, lang, split=split, trust_remote_code=True)


def phonemize_greek(text: str) -> str:
    """Convert Greek text to IPA phonemes using espeak-ng."""
    text = re.sub(r'[^\w\s\u0370-\u03FF\u1F00-\u1FFF]', '', text)
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
    except Exception:
        return ""


def normalize_audio(wav, sr, target_sr=22050):
    """Normalize audio to target sample rate."""
    import torchaudio.transforms as T
    
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    if sr != target_sr:
        resampler = T.Resample(sr, target_sr)
        wav = resampler(wav)
    
    wav = wav / (wav.abs().max() + 1e-8) * 0.95
    
    return wav, target_sr


def download_fleurs(output_dir: Path, max_samples: Optional[int] = None) -> list:
    """Download and prepare Google Fleurs Greek."""
    print("\n📥 Downloading Google Fleurs (Greek)...")
    
    try:
        print("   Loading google/fleurs...")
        ds = _load_dataset_compat("google/fleurs", "el_gr", split="train")
        print(f"   ✅ Loaded {len(ds)} samples")
    except Exception as e:
        print(f"❌ Failed to load Fleurs: {e}")
        print("   Try: pip install 'datasets>=2.14.0,<3.0.0'")
        return []
    
    audio_dir = output_dir / "fleurs" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    total = min(len(ds), max_samples) if max_samples else len(ds)
    
    for idx, sample in enumerate(tqdm(ds, total=total, desc="Processing")):
        if max_samples and idx >= max_samples:
            break
        
        if not sample.get("transcription") or not sample.get("audio"):
            continue
        
        text = sample["transcription"].strip()
        if len(text) < 5:
            continue
        
        # Process audio
        audio_array = sample["audio"]["array"]
        orig_sr = sample["audio"]["sampling_rate"]
        
        import numpy as np
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
        else:
            audio_tensor = torch.tensor(audio_array).float().unsqueeze(0)
        
        wav, sr = normalize_audio(audio_tensor, orig_sr)
        
        # Check duration
        duration = wav.shape[1] / sr
        if duration < 1.0 or duration > 15.0:
            continue
        
        # Phonemize
        phonemes = phonemize_greek(text)
        if not phonemes:
            continue
        
        # Save
        output_path = audio_dir / f"fleurs_{idx:06d}.wav"
        save_audio(str(output_path), wav, sr)
        
        manifest.append({
            "audio": str(output_path),
            "text": text,
            "phonemes": phonemes,
            "lang": "el",
            "duration": round(duration, 2),
            "source": "fleurs"
        })
    
    print(f"   ✅ Processed {len(manifest)} samples")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download Greek TTS datasets (Fleurs)")
    parser.add_argument("--output_dir", type=str, default="data/el",
                        help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples (default: all)")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation split ratio")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Fleurs
    manifest = download_fleurs(output_dir, args.max_samples)
    
    if not manifest:
        print("\n❌ No data was processed")
        return
    
    # Shuffle and split
    import random
    random.shuffle(manifest)
    
    n_val = int(len(manifest) * args.val_ratio)
    val_manifest = manifest[:n_val]
    train_manifest = manifest[n_val:]
    
    # Save manifests
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    
    with open(manifests_dir / "train_manifest_el.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2, ensure_ascii=False)
    
    with open(manifests_dir / "val_manifest_el.json", "w", encoding="utf-8") as f:
        json.dump(val_manifest, f, indent=2, ensure_ascii=False)
    
    # Print summary
    total_duration = sum(s.get("duration", 0) for s in manifest)
    
    print("\n" + "="*50)
    print("📊 Download Summary")
    print("="*50)
    print(f"Total samples:     {len(manifest)}")
    print(f"Training samples:  {len(train_manifest)}")
    print(f"Validation samples: {len(val_manifest)}")
    print(f"Total duration:    {total_duration/3600:.2f} hours")
    print(f"\n📁 Output: {output_dir}")
    print(f"   Manifests: {manifests_dir}")
    
    print("\n🚀 To create archive for upload:")
    print(f"   tar -czvf greek_data.tar.gz -C {output_dir.parent} {output_dir.name}")


if __name__ == "__main__":
    main()
