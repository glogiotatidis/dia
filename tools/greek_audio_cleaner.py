#!/usr/bin/env python3
"""
Greek Audio Cleaning and Preprocessing Tool

This tool processes raw Greek audio data for TTS training:
- Audio normalization and volume leveling
- Silence trimming
- Resampling to target sample rate
- Duration filtering
- Quality checks

Usage:
    python tools/greek_audio_cleaner.py \
        --input_dir raw_audio/ \
        --output_dir cleaned_audio/ \
        --sample_rate 22050
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


class GreekAudioCleaner:
    """Audio preprocessing pipeline for Greek TTS training."""
    
    def __init__(
        self,
        target_sr: int = 22050,
        target_db: float = -20.0,
        min_duration: float = 1.0,
        max_duration: float = 15.0,
        trim_silence: bool = True,
        silence_threshold_db: float = -40.0,
    ):
        self.target_sr = target_sr
        self.target_db = target_db
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.trim_silence = trim_silence
        self.silence_threshold_db = silence_threshold_db
    
    def load_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and return waveform and sample rate."""
        try:
            wav, sr = torchaudio.load(path)
            return wav, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")
    
    def to_mono(self, wav: torch.Tensor) -> torch.Tensor:
        """Convert stereo to mono."""
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav
    
    def resample(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if orig_sr != self.target_sr:
            resampler = T.Resample(orig_sr, self.target_sr)
            wav = resampler(wav)
        return wav
    
    def normalize_volume(self, wav: torch.Tensor) -> torch.Tensor:
        """Normalize audio volume to target dB level."""
        vol_transform = T.Vol(self.target_db, gain_type='db')
        return vol_transform(wav)
    
    def trim_silence_vad(self, wav: torch.Tensor) -> torch.Tensor:
        """Trim silence from beginning and end using VAD."""
        if not self.trim_silence:
            return wav
        
        try:
            # Use torchaudio's VAD
            vad = T.Vad(sample_rate=self.target_sr)
            trimmed = vad(wav)
            
            # If VAD removed too much, return original
            if trimmed.shape[1] < self.min_duration * self.target_sr:
                return wav
            
            return trimmed
        except Exception:
            # If VAD fails, try simple threshold-based trimming
            return self._simple_trim(wav)
    
    def _simple_trim(self, wav: torch.Tensor) -> torch.Tensor:
        """Simple threshold-based silence trimming."""
        # Calculate RMS energy
        energy = wav.pow(2).mean(dim=0).sqrt()
        threshold = 10 ** (self.silence_threshold_db / 20)
        
        # Find non-silent regions
        non_silent = energy > threshold
        
        if not non_silent.any():
            return wav
        
        # Get first and last non-silent indices
        indices = torch.where(non_silent)[0]
        start = max(0, indices[0].item() - int(0.1 * self.target_sr))  # 100ms padding
        end = min(wav.shape[1], indices[-1].item() + int(0.1 * self.target_sr))
        
        return wav[:, start:end]
    
    def check_duration(self, wav: torch.Tensor) -> Tuple[bool, float]:
        """Check if audio duration is within acceptable range."""
        duration = wav.shape[1] / self.target_sr
        valid = self.min_duration <= duration <= self.max_duration
        return valid, duration
    
    def check_quality(self, wav: torch.Tensor) -> Tuple[bool, dict]:
        """Basic audio quality checks."""
        stats = {}
        
        # Check for clipping
        max_val = wav.abs().max().item()
        stats["max_amplitude"] = max_val
        stats["clipping"] = max_val > 0.99
        
        # Check for DC offset
        mean_val = wav.mean().item()
        stats["dc_offset"] = abs(mean_val)
        stats["has_dc_offset"] = abs(mean_val) > 0.01
        
        # Check for silence (too quiet)
        rms = wav.pow(2).mean().sqrt().item()
        stats["rms"] = rms
        stats["too_quiet"] = rms < 0.001
        
        # Overall quality
        quality_ok = not stats["clipping"] and not stats["too_quiet"]
        
        return quality_ok, stats
    
    def remove_dc_offset(self, wav: torch.Tensor) -> torch.Tensor:
        """Remove DC offset from audio."""
        return wav - wav.mean()
    
    def process(self, input_path: str, output_path: str) -> Tuple[bool, dict]:
        """
        Process a single audio file through the entire pipeline.
        
        Returns:
            Tuple of (success, info_dict)
        """
        info = {"input": input_path, "output": output_path}
        
        try:
            # Load
            wav, sr = self.load_audio(input_path)
            info["original_sr"] = sr
            info["original_duration"] = wav.shape[1] / sr
            
            # Convert to mono
            wav = self.to_mono(wav)
            
            # Resample
            wav = self.resample(wav, sr)
            
            # Remove DC offset
            wav = self.remove_dc_offset(wav)
            
            # Normalize volume
            wav = self.normalize_volume(wav)
            
            # Trim silence
            wav = self.trim_silence_vad(wav)
            
            # Check duration
            duration_ok, duration = self.check_duration(wav)
            info["duration"] = duration
            
            if not duration_ok:
                info["reason"] = f"Duration {duration:.2f}s out of range [{self.min_duration}, {self.max_duration}]"
                return False, info
            
            # Quality check
            quality_ok, quality_stats = self.check_quality(wav)
            info["quality"] = quality_stats
            
            if not quality_ok:
                info["reason"] = f"Quality check failed: {quality_stats}"
                return False, info
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, wav, self.target_sr)
            
            info["success"] = True
            return True, info
            
        except Exception as e:
            info["reason"] = str(e)
            return False, info


def process_directory(
    input_dir: str,
    output_dir: str,
    cleaner: GreekAudioCleaner,
    extensions: list = [".wav", ".mp3", ".flac", ".ogg"],
) -> dict:
    """Process all audio files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_path.glob(f"**/*{ext}"))
    
    print(f"Found {len(audio_files)} audio files")
    
    results = {
        "total": len(audio_files),
        "success": 0,
        "failed": 0,
        "total_duration": 0.0,
        "files": [],
    }
    
    for audio_file in tqdm(audio_files, desc="Processing"):
        # Generate output path maintaining directory structure
        rel_path = audio_file.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix(".wav")
        
        success, info = cleaner.process(str(audio_file), str(out_file))
        
        if success:
            results["success"] += 1
            results["total_duration"] += info.get("duration", 0)
        else:
            results["failed"] += 1
        
        results["files"].append(info)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Greek Audio Cleaning Tool")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory containing raw audio")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for cleaned audio")
    parser.add_argument("--sample_rate", type=int, default=22050,
                        help="Target sample rate")
    parser.add_argument("--target_db", type=float, default=-20.0,
                        help="Target volume in dB")
    parser.add_argument("--min_duration", type=float, default=1.0,
                        help="Minimum audio duration in seconds")
    parser.add_argument("--max_duration", type=float, default=15.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--no_trim", action="store_true",
                        help="Disable silence trimming")
    parser.add_argument("--report_path", type=str, default=None,
                        help="Path to save processing report JSON")
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = GreekAudioCleaner(
        target_sr=args.sample_rate,
        target_db=args.target_db,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        trim_silence=not args.no_trim,
    )
    
    # Process directory
    results = process_directory(args.input_dir, args.output_dir, cleaner)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 Processing Summary")
    print("="*50)
    print(f"Total files:     {results['total']}")
    print(f"Successful:      {results['success']}")
    print(f"Failed:          {results['failed']}")
    print(f"Total duration:  {results['total_duration']/3600:.2f} hours")
    print(f"Output dir:      {args.output_dir}")
    
    # Save report if requested
    if args.report_path:
        with open(args.report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Report saved:    {args.report_path}")
    
    # Print failed files
    failed = [f for f in results["files"] if not f.get("success")]
    if failed and len(failed) <= 10:
        print("\n❌ Failed files:")
        for f in failed:
            print(f"   {f['input']}: {f.get('reason', 'Unknown')}")


if __name__ == "__main__":
    main()
