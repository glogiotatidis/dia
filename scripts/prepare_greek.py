#!/usr/bin/env python3
"""
Greek (el) Language Data Preparation Script for DIA Multilingual TTS

This script prepares Greek speech data for training the DIA TTS model.
It supports multiple data sources and handles:
- Downloading from HuggingFace datasets (Common Voice, etc.)
- Audio preprocessing (resampling, normalization, trimming)
- Phonemization using espeak-ng
- Manifest generation in the required format

Usage:
    python scripts/prepare_greek.py --source commonvoice --output_dir data/el
    python scripts/prepare_greek.py --source local --audio_dir /path/to/audio --output_dir data/el
"""

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Optional

import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


def check_espeak_greek_support():
    """Verify that espeak-ng supports Greek."""
    try:
        result = subprocess.run(
            ["espeak-ng", "--voices=el"],
            capture_output=True,
            text=True
        )
        if "el" in result.stdout or "greek" in result.stdout.lower():
            print("✅ espeak-ng Greek support verified")
            return True
        else:
            print("❌ Greek voice not found in espeak-ng")
            print("Install with: apt-get install espeak-ng espeak-ng-data")
            return False
    except FileNotFoundError:
        print("❌ espeak-ng not found. Install it first:")
        print("   macOS: brew install espeak-ng")
        print("   Ubuntu: apt-get install espeak-ng")
        return False


def phonemize_greek(text: str) -> str:
    """
    Convert Greek text to IPA phonemes using espeak-ng.
    
    Args:
        text: Greek text string
        
    Returns:
        IPA phoneme string
    """
    # Clean text - remove special characters but keep Greek letters
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
        # Clean up phoneme output
        phonemes = re.sub(r'\s+', ' ', phonemes)
        return phonemes
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        print(f"Warning: Failed to phonemize '{text[:50]}...': {e}")
        return ""


def normalize_audio(
    input_path: str,
    output_path: str,
    target_sr: int = 22050,
    target_db: float = -20.0,
    trim_silence: bool = True
) -> bool:
    """
    Normalize and preprocess audio file.
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save processed audio
        target_sr: Target sample rate (22050 for DIA)
        target_db: Target loudness in dB
        trim_silence: Whether to trim silence from beginning/end
        
    Returns:
        True if successful, False otherwise
    """
    try:
        wav, sr = torchaudio.load(input_path)
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            wav = resampler(wav)
        
        # Normalize volume
        vol_transform = T.Vol(target_db, gain_type='db')
        wav = vol_transform(wav)
        
        # Trim silence (optional)
        if trim_silence:
            try:
                vad = T.Vad(sample_rate=target_sr)
                wav = vad(wav)
            except Exception:
                pass  # VAD can fail on some audio, skip if it does
        
        # Ensure audio is not too short (minimum 0.5 seconds)
        min_samples = int(0.5 * target_sr)
        if wav.shape[1] < min_samples:
            return False
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, wav, target_sr)
        return True
        
    except Exception as e:
        print(f"Warning: Failed to process {input_path}: {e}")
        return False


def get_audio_duration(path: str, sample_rate: int = 22050) -> float:
    """Get audio duration in seconds."""
    try:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        return 0.0


def download_commonvoice_greek(output_dir: str, max_samples: Optional[int] = None) -> list:
    """
    Download and prepare Greek data from Mozilla Common Voice.
    
    Args:
        output_dir: Directory to save processed data
        max_samples: Maximum number of samples to process (None for all)
        
    Returns:
        List of manifest entries
    """
    from datasets import load_dataset
    
    print("📥 Downloading Greek Common Voice dataset...")
    print("Note: This requires accepting the dataset license on HuggingFace")
    
    try:
        # Try Common Voice 17.0 first, fall back to 16.0, then 13.0
        for version in ["17.0", "16.0", "13.0"]:
            try:
                ds = load_dataset(
                    f"mozilla-foundation/common_voice_{version.replace('.', '_')}",
                    "el",
                    split="train",
                    trust_remote_code=True
                )
                print(f"✅ Loaded Common Voice {version}")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Could not load Common Voice dataset")
            
    except Exception as e:
        print(f"❌ Failed to load Common Voice: {e}")
        print("\nTo use Common Voice, you need to:")
        print("1. Create a HuggingFace account")
        print("2. Accept the dataset license at:")
        print("   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0")
        print("3. Login with: huggingface-cli login")
        return []
    
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    total = min(len(ds), max_samples) if max_samples else len(ds)
    
    print(f"🔄 Processing {total} samples...")
    
    for idx, sample in enumerate(tqdm(ds, total=total)):
        if max_samples and idx >= max_samples:
            break
            
        if not sample.get("sentence") or not sample.get("audio"):
            continue
        
        text = sample["sentence"].strip()
        if len(text) < 5:  # Skip very short utterances
            continue
            
        # Get audio data
        audio_array = sample["audio"]["array"]
        orig_sr = sample["audio"]["sampling_rate"]
        
        # Save original audio temporarily
        temp_path = audio_dir / f"temp_{idx}.wav"
        output_path = audio_dir / f"el_{idx:06d}.wav"
        
        import torch
        import numpy as np
        
        # Convert to tensor
        if isinstance(audio_array, np.ndarray):
            audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
        else:
            audio_tensor = torch.tensor(audio_array).float().unsqueeze(0)
        
        torchaudio.save(str(temp_path), audio_tensor, orig_sr)
        
        # Normalize audio
        if not normalize_audio(str(temp_path), str(output_path)):
            temp_path.unlink(missing_ok=True)
            continue
        
        temp_path.unlink(missing_ok=True)
        
        # Check duration (skip if too long or too short)
        duration = get_audio_duration(str(output_path))
        if duration < 1.0 or duration > 15.0:
            output_path.unlink(missing_ok=True)
            continue
        
        # Phonemize
        phonemes = phonemize_greek(text)
        if not phonemes:
            output_path.unlink(missing_ok=True)
            continue
        
        manifest.append({
            "audio": str(output_path),
            "text": text,
            "phonemes": phonemes,
            "lang": "el",
            "duration": round(duration, 2)
        })
    
    print(f"✅ Processed {len(manifest)} valid samples")
    return manifest


def prepare_local_data(
    audio_dir: str,
    transcript_file: str,
    output_dir: str,
    max_samples: Optional[int] = None
) -> list:
    """
    Prepare Greek data from local audio files and transcripts.
    
    Expected transcript format (TSV or JSON):
    - TSV: filename<tab>text
    - JSON: [{"audio": "path", "text": "text"}, ...]
    
    Args:
        audio_dir: Directory containing audio files
        transcript_file: Path to transcript file
        output_dir: Directory to save processed data
        max_samples: Maximum number of samples
        
    Returns:
        List of manifest entries
    """
    audio_dir = Path(audio_dir)
    output_audio_dir = Path(output_dir) / "audio"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    
    # Load transcripts
    transcripts = {}
    transcript_path = Path(transcript_file)
    
    if transcript_path.suffix == ".json":
        with open(transcript_path) as f:
            data = json.load(f)
            for item in data:
                audio_name = Path(item["audio"]).name
                transcripts[audio_name] = item["text"]
    else:  # TSV format
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    transcripts[parts[0]] = parts[1]
    
    manifest = []
    audio_files = list(audio_dir.glob("**/*.wav")) + list(audio_dir.glob("**/*.mp3"))
    
    if max_samples:
        audio_files = audio_files[:max_samples]
    
    print(f"🔄 Processing {len(audio_files)} local audio files...")
    
    for idx, audio_path in enumerate(tqdm(audio_files)):
        audio_name = audio_path.name
        
        if audio_name not in transcripts:
            continue
        
        text = transcripts[audio_name].strip()
        if len(text) < 5:
            continue
        
        output_path = output_audio_dir / f"el_local_{idx:06d}.wav"
        
        if not normalize_audio(str(audio_path), str(output_path)):
            continue
        
        duration = get_audio_duration(str(output_path))
        if duration < 1.0 or duration > 15.0:
            output_path.unlink(missing_ok=True)
            continue
        
        phonemes = phonemize_greek(text)
        if not phonemes:
            output_path.unlink(missing_ok=True)
            continue
        
        manifest.append({
            "audio": str(output_path),
            "text": text,
            "phonemes": phonemes,
            "lang": "el",
            "duration": round(duration, 2)
        })
    
    print(f"✅ Processed {len(manifest)} valid samples from local data")
    return manifest


def build_phoneme_vocab(manifest: list) -> dict:
    """Build phoneme vocabulary from manifest."""
    chars = Counter()
    for sample in manifest:
        chars.update(sample["phonemes"])
    
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, char in enumerate(sorted(chars.keys())):
        vocab[char] = i + 10  # Reserve 0-9 for special tokens
    
    return vocab


def split_manifest(manifest: list, val_ratio: float = 0.05, test_ratio: float = 0.05):
    """Split manifest into train/val/test sets."""
    import random
    random.shuffle(manifest)
    
    n = len(manifest)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    
    test = manifest[:n_test]
    val = manifest[n_test:n_test + n_val]
    train = manifest[n_test + n_val:]
    
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description="Prepare Greek data for DIA TTS training")
    parser.add_argument("--source", choices=["commonvoice", "local"], default="commonvoice",
                        help="Data source")
    parser.add_argument("--output_dir", type=str, default="data/el",
                        help="Output directory for processed data")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Local audio directory (for --source local)")
    parser.add_argument("--transcript_file", type=str, default=None,
                        help="Transcript file path (for --source local)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.05,
                        help="Test set ratio")
    
    args = parser.parse_args()
    
    # Check espeak-ng
    if not check_espeak_greek_support():
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data based on source
    if args.source == "commonvoice":
        manifest = download_commonvoice_greek(str(output_dir), args.max_samples)
    else:
        if not args.audio_dir or not args.transcript_file:
            print("❌ --audio_dir and --transcript_file required for local source")
            return
        manifest = prepare_local_data(
            args.audio_dir, args.transcript_file, str(output_dir), args.max_samples
        )
    
    if not manifest:
        print("❌ No data processed")
        return
    
    # Split data
    train, val, test = split_manifest(manifest, args.val_ratio, args.test_ratio)
    
    # Save manifests
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(exist_ok=True)
    
    with open(manifests_dir / "train_manifest_el.json", "w", encoding="utf-8") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    
    with open(manifests_dir / "val_manifest_el.json", "w", encoding="utf-8") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    
    with open(manifests_dir / "test_manifest_el.json", "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2, ensure_ascii=False)
    
    # Build and save phoneme vocab
    phoneme_vocab = build_phoneme_vocab(manifest)
    with open(manifests_dir / "phoneme_vocab_el.json", "w", encoding="utf-8") as f:
        json.dump(phoneme_vocab, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "="*50)
    print("📊 Greek Data Preparation Complete")
    print("="*50)
    print(f"Total samples:     {len(manifest)}")
    print(f"Training samples:  {len(train)}")
    print(f"Validation samples: {len(val)}")
    print(f"Test samples:      {len(test)}")
    print(f"Phoneme vocab size: {len(phoneme_vocab)}")
    
    total_duration = sum(s.get("duration", 0) for s in manifest)
    print(f"Total audio:       {total_duration/3600:.2f} hours")
    
    print(f"\n📁 Output saved to: {output_dir}")
    print(f"   - Audio:     {output_dir}/audio/")
    print(f"   - Manifests: {manifests_dir}/")
    
    print("\n🚀 Next steps:")
    print("1. Review the generated manifests")
    print("2. Run training with:")
    print(f"   python scripts/train_greek.py --manifest {manifests_dir}/train_manifest_el.json")


if __name__ == "__main__":
    main()
