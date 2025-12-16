#!/usr/bin/env python3
"""
Download and prepare multiple Greek speech datasets for TTS training.

Supported datasets:
- Mozilla Common Voice (el) - requires HuggingFace login and license acceptance
- Google Fleurs (el_gr) - freely available
- CSS10 Greek (manual download required)

Usage:
    python scripts/download_greek_datasets.py --datasets fleurs --output_dir data/el
    python scripts/download_greek_datasets.py --datasets commonvoice fleurs --output_dir data/el
"""

import argparse
import json
import os
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
    # Ensure wav is 2D (channels, samples) and convert to numpy
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    # soundfile expects (samples, channels)
    wav_np = wav.squeeze(0).numpy() if wav.shape[0] == 1 else wav.T.numpy()
    sf.write(path, wav_np, sr)


def _load_dataset_compat(dataset_name: str, lang: str, split: str = "train"):
    """Load dataset with compatibility for different datasets library versions."""
    from datasets import load_dataset
    import datasets
    
    # Check datasets version
    major_version = int(datasets.__version__.split('.')[0])
    
    if major_version >= 3:
        # datasets 3.x+ doesn't support trust_remote_code
        return load_dataset(dataset_name, lang, split=split)
    else:
        # datasets 2.x requires trust_remote_code for custom loading scripts
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
    
    # Normalize volume
    wav = wav / (wav.abs().max() + 1e-8) * 0.95
    
    return wav, target_sr


def download_commonvoice(output_dir: Path, max_samples: Optional[int] = None) -> list:
    """Download and prepare Common Voice Greek."""
    print("\n📥 Downloading Mozilla Common Voice (Greek)...")
    print("   This requires accepting the license at:")
    print("   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
    
    ds = None
    
    # Try different Common Voice versions (newest first)
    cv_versions = [
        ("mozilla-foundation/common_voice_17_0", "el"),
        ("mozilla-foundation/common_voice_16_1", "el"),
        ("mozilla-foundation/common_voice_16_0", "el"),
        ("mozilla-foundation/common_voice_13_0", "el"),
    ]
    
    for dataset_name, lang in cv_versions:
        try:
            print(f"   Trying {dataset_name}...")
            ds = _load_dataset_compat(dataset_name, lang, split="train")
            print(f"   ✅ Loaded {dataset_name}")
            break
        except Exception as e:
            print(f"   ⚠️  {dataset_name} failed: {str(e)[:100]}")
            continue
    
    if ds is None:
        print("❌ Failed to load Common Voice")
        print("   Make sure you have:")
        print("   1. Logged in: huggingface-cli login")
        print("   2. Accepted the license at the dataset page")
        return []
    
    audio_dir = output_dir / "commonvoice" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    total = min(len(ds), max_samples) if max_samples else len(ds)
    
    for idx, sample in enumerate(tqdm(ds, total=total, desc="Common Voice")):
        if max_samples and idx >= max_samples:
            break
        
        if not sample.get("sentence") or not sample.get("audio"):
            continue
        
        text = sample["sentence"].strip()
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
        output_path = audio_dir / f"cv_{idx:06d}.wav"
        save_audio(str(output_path), wav, sr)
        
        manifest.append({
            "audio": str(output_path),
            "text": text,
            "phonemes": phonemes,
            "lang": "el",
            "duration": round(duration, 2),
            "source": "commonvoice"
        })
    
    print(f"   ✅ Processed {len(manifest)} samples from Common Voice")
    return manifest


def download_fleurs(output_dir: Path, max_samples: Optional[int] = None) -> list:
    """Download and prepare Google Fleurs Greek."""
    print("\n📥 Downloading Google Fleurs (Greek)...")
    
    ds = None
    
    # Try different Fleurs sources
    fleurs_sources = [
        ("google/fleurs", "el_gr"),
    ]
    
    for dataset_name, lang in fleurs_sources:
        try:
            print(f"   Trying {dataset_name}...")
            ds = _load_dataset_compat(dataset_name, lang, split="train")
            print(f"   ✅ Loaded {dataset_name}")
            break
        except Exception as e:
            print(f"   ⚠️  {dataset_name} failed: {str(e)[:100]}")
            continue
    
    if ds is None:
        print("❌ Failed to load Fleurs")
        print("   Try: pip install 'datasets>=2.14.0,<3.0.0'")
        return []
    
    audio_dir = output_dir / "fleurs" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    total = min(len(ds), max_samples) if max_samples else len(ds)
    
    for idx, sample in enumerate(tqdm(ds, total=total, desc="Fleurs")):
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
    
    print(f"   ✅ Processed {len(manifest)} samples from Fleurs")
    return manifest


def download_css10(output_dir: Path, css10_path: str) -> list:
    """Prepare CSS10 Greek from local download."""
    print("\n📥 Processing CSS10 Greek...")
    print(f"   Source: {css10_path}")
    
    css10_dir = Path(css10_path)
    if not css10_dir.exists():
        print("❌ CSS10 directory not found")
        print("   Download from: https://github.com/Kyubyong/css10")
        print("   Or: https://www.kaggle.com/datasets/bryanpark/css10")
        return []
    
    audio_dir = output_dir / "css10" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Find transcript file
    transcript_file = css10_dir / "transcript.txt"
    if not transcript_file.exists():
        print(f"❌ transcript.txt not found in {css10_dir}")
        return []
    
    manifest = []
    
    with open(transcript_file, encoding="utf-8") as f:
        lines = f.readlines()
    
    for idx, line in enumerate(tqdm(lines, desc="CSS10")):
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        
        audio_name = parts[0]
        text = parts[1].strip()
        
        if len(text) < 5:
            continue
        
        # Find audio file
        audio_path = css10_dir / audio_name
        if not audio_path.exists():
            audio_path = css10_dir / f"{audio_name}.wav"
        if not audio_path.exists():
            continue
        
        # Process audio
        try:
            wav, orig_sr = torchaudio.load(str(audio_path))
            wav, sr = normalize_audio(wav, orig_sr)
        except Exception:
            continue
        
        # Check duration
        duration = wav.shape[1] / sr
        if duration < 1.0 or duration > 15.0:
            continue
        
        # Phonemize
        phonemes = phonemize_greek(text)
        if not phonemes:
            continue
        
        # Save
        output_path = audio_dir / f"css10_{idx:06d}.wav"
        save_audio(str(output_path), wav, sr)
        
        manifest.append({
            "audio": str(output_path),
            "text": text,
            "phonemes": phonemes,
            "lang": "el",
            "duration": round(duration, 2),
            "source": "css10"
        })
    
    print(f"   ✅ Processed {len(manifest)} samples from CSS10")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Download Greek TTS datasets")
    parser.add_argument("--datasets", nargs="+", 
                        choices=["commonvoice", "fleurs", "css10", "all"],
                        default=["fleurs"],
                        help="Datasets to download (fleurs is freely available, commonvoice requires HF login)")
    parser.add_argument("--output_dir", type=str, default="data/el",
                        help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per dataset (for testing)")
    parser.add_argument("--css10_path", type=str, default=None,
                        help="Path to CSS10 Greek directory (if using css10)")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation split ratio")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["commonvoice", "fleurs"]
        if args.css10_path:
            datasets.append("css10")
    
    all_manifest = []
    
    # Download each dataset
    if "commonvoice" in datasets:
        manifest = download_commonvoice(output_dir, args.max_samples)
        all_manifest.extend(manifest)
    
    if "fleurs" in datasets:
        manifest = download_fleurs(output_dir, args.max_samples)
        all_manifest.extend(manifest)
    
    if "css10" in datasets:
        if not args.css10_path:
            print("⚠️  CSS10 requires --css10_path argument")
        else:
            manifest = download_css10(output_dir, args.css10_path)
            all_manifest.extend(manifest)
    
    if not all_manifest:
        print("\n❌ No data was processed")
        return
    
    # Shuffle and split
    import random
    random.shuffle(all_manifest)
    
    n_val = int(len(all_manifest) * args.val_ratio)
    val_manifest = all_manifest[:n_val]
    train_manifest = all_manifest[n_val:]
    
    # Save manifests
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    
    with open(manifests_dir / "train_manifest_el.json", "w", encoding="utf-8") as f:
        json.dump(train_manifest, f, indent=2, ensure_ascii=False)
    
    with open(manifests_dir / "val_manifest_el.json", "w", encoding="utf-8") as f:
        json.dump(val_manifest, f, indent=2, ensure_ascii=False)
    
    # Print summary
    total_duration = sum(s.get("duration", 0) for s in all_manifest)
    
    print("\n" + "="*50)
    print("📊 Download Summary")
    print("="*50)
    print(f"Total samples:     {len(all_manifest)}")
    print(f"Training samples:  {len(train_manifest)}")
    print(f"Validation samples: {len(val_manifest)}")
    print(f"Total duration:    {total_duration/3600:.2f} hours")
    
    # Per-source breakdown
    sources = {}
    for s in all_manifest:
        src = s.get("source", "unknown")
        if src not in sources:
            sources[src] = {"count": 0, "duration": 0}
        sources[src]["count"] += 1
        sources[src]["duration"] += s.get("duration", 0)
    
    print("\nPer-source breakdown:")
    for src, stats in sources.items():
        print(f"  {src}: {stats['count']} samples, {stats['duration']/3600:.2f}h")
    
    print(f"\n📁 Output: {output_dir}")
    print(f"   Manifests: {manifests_dir}")
    
    print("\n🚀 Next step:")
    print(f"   python scripts/train_greek.py --manifest {manifests_dir}/train_manifest_el.json")


if __name__ == "__main__":
    main()
