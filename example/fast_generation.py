#!/usr/bin/env python3
"""
Example: Fast TTS generation with optimizations.

This script demonstrates how to use various optimization techniques
to speed up Dia TTS inference on GPU or CPU.

Usage:
    # GPU inference
    python example/fast_generation.py --text "Hello, how are you today?"
    
    # CPU inference with quantization
    python example/fast_generation.py --cpu --quantize --text "Hello world"
    
Optimization levels (GPU):
    --level 1: Basic (torch.compile)
    --level 2: + Mixed precision (FP16/BF16)
    --level 3: + All optimizations

CPU options:
    --cpu: Force CPU inference
    --quantize: Apply INT8 quantization (2-4x speedup)
    --threads N: Number of CPU threads to use
"""

import argparse
import time
import torch
import numpy as np
import soundfile as sf
from pathlib import Path


def benchmark_generation(generate_fn, text: str, warmup: int = 2, runs: int = 5, device_type: str = "cuda"):
    """Benchmark a generation function."""
    # Warmup runs (important for torch.compile)
    print(f"Warming up with {warmup} runs...")
    for _ in range(warmup):
        _ = generate_fn(text, max_tokens=500)
    
    # Synchronize GPU if applicable
    if device_type == "cuda":
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    print(f"Running {runs} timed generations...")
    for i in range(runs):
        start = time.perf_counter()
        audio = generate_fn(text, max_tokens=500)
        if device_type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "audio_length": len(audio) / 44100,  # seconds at 44.1kHz
    }


def setup_gpu_optimizations(level: int):
    """Apply GPU-specific optimizations."""
    optimizations = []
    
    if level >= 1:
        optimizations.append("torch.compile (reduce-overhead mode)")
    
    if level >= 2:
        optimizations.append("Mixed precision (autocast)")
        optimizations.append("Optimized sampling")
    
    if level >= 3:
        torch.backends.cudnn.benchmark = True
        optimizations.append("cuDNN benchmark mode")
        
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
            optimizations.append("High-precision matmul")
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        optimizations.append("TensorFloat-32 enabled")
    
    return optimizations


def setup_cpu_optimizations(threads: int | None, quantize: bool):
    """Apply CPU-specific optimizations."""
    from dia.fast_inference import optimize_for_cpu, check_cpu_features
    
    # Show CPU features
    print("\nChecking CPU capabilities...")
    check_cpu_features()
    
    # Apply CPU optimizations
    optimize_for_cpu(num_threads=threads)
    
    optimizations = [
        f"Thread optimization ({torch.get_num_threads()} threads)",
        "MKL-DNN backend",
    ]
    
    if quantize:
        optimizations.append("INT8 dynamic quantization")
    
    return optimizations


def main():
    parser = argparse.ArgumentParser(description="Fast Dia TTS Generation")
    parser.add_argument("--text", type=str, default="[S1] Hello, this is a test of the Dia text to speech system.")
    parser.add_argument("--output", type=str, default="output_fast.wav")
    parser.add_argument("--model", type=str, default=None, help="Path to local model checkpoint")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    
    # GPU options
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3],
                        help="GPU optimization level (1=compile, 2=+fp16, 3=+all)")
    
    # CPU options
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--quantize", action="store_true", help="Apply INT8 quantization (CPU)")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads")
    
    # Advanced options
    parser.add_argument("--reduced-cfg", action="store_true", 
                        help="Use reduced CFG (only for first 50 tokens)")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="CFG scale")
    
    args = parser.parse_args()
    
    # Determine device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"{'='*60}")
    print(f"Dia TTS Fast Inference")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
        print(f"BF16 support: {'Yes' if torch.cuda.is_bf16_supported() else 'No'}")
    else:
        import platform
        print(f"CPU: {platform.processor() or 'Unknown'}")
    
    # Load model
    from dia.model import Dia
    
    print(f"\nLoading model to {device}...")
    load_start = time.perf_counter()
    
    if args.model:
        config_path = Path(args.model).parent / "config.json"
        model = Dia.from_local(str(config_path), args.model, device=device)
    else:
        model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
    
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.1f}s")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Apply optimizations based on device
    print(f"\nApplying optimizations...")
    
    if device.type == "cuda":
        optimizations = setup_gpu_optimizations(args.level)
        
        if args.reduced_cfg:
            from dia.fast_inference import create_reduced_cfg_generate
            generate_fn = create_reduced_cfg_generate(model, cfg_steps=50)
            optimizations.append("Reduced CFG (first 50 tokens only)")
        elif args.level >= 2:
            from dia.fast_inference import create_fast_generate
            generate_fn = create_fast_generate(model)
        else:
            def generate_fn(text, **kwargs):
                return model.generate(text, use_torch_compile=(args.level >= 1), 
                                       cfg_scale=args.cfg_scale, **kwargs)
    else:
        optimizations = setup_cpu_optimizations(args.threads, args.quantize)
        
        from dia.fast_inference import create_cpu_generate
        generate_fn = create_cpu_generate(
            model, 
            quantize=args.quantize,
            use_compile=True
        )
    
    for opt in optimizations:
        print(f"  ✓ {opt}")
    
    # Generate audio
    text_display = f"\"{args.text[:50]}...\"" if len(args.text) > 50 else f"\"{args.text}\""
    print(f"\nGenerating audio for: {text_display}")
    
    if args.benchmark:
        results = benchmark_generation(generate_fn, args.text, device_type=device.type)
        print(f"\n{'='*60}")
        print("Benchmark Results:")
        print(f"{'='*60}")
        print(f"  Mean time:        {results['mean']:.2f}s ± {results['std']:.2f}s")
        print(f"  Min/Max:          {results['min']:.2f}s / {results['max']:.2f}s")
        print(f"  Audio length:     {results['audio_length']:.2f}s")
        print(f"  Real-time factor: {results['mean'] / results['audio_length']:.2f}x")
        if results['mean'] < results['audio_length']:
            print(f"  ⚡ Faster than real-time!")
        print(f"{'='*60}")
    else:
        start = time.perf_counter()
        audio = generate_fn(args.text, cfg_scale=args.cfg_scale)
        elapsed = time.perf_counter() - start
        
        # Save audio
        sf.write(args.output, audio, 44100)
        audio_len = len(audio) / 44100
        rtf = elapsed / audio_len
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"{'='*60}")
        print(f"  Output:           {args.output}")
        print(f"  Generation time:  {elapsed:.2f}s")
        print(f"  Audio length:     {audio_len:.2f}s")
        print(f"  Real-time factor: {rtf:.2f}x")
        if rtf < 1.0:
            print(f"  ⚡ {1/rtf:.1f}x faster than real-time!")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
