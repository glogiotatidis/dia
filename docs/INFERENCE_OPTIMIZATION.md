# Dia TTS Inference Optimization Guide

This guide covers techniques to speed up Dia TTS audio generation on GPUs and CPUs.

## Quick Start

### GPU (Recommended)

```python
from dia.model import Dia
from dia.fast_inference import create_fast_generate

# Load model
model = Dia.from_pretrained("nari-labs/Dia-1.6B", device="cuda")

# Create optimized generate function
fast_generate = create_fast_generate(model)

# Generate audio (2-3x faster than default)
audio = fast_generate("[S1] Hello world!")
```

### CPU

```python
from dia.model import Dia
from dia.fast_inference import optimize_for_cpu, create_cpu_generate

# Configure CPU optimizations
optimize_for_cpu(num_threads=8)

# Load model
model = Dia.from_pretrained("nari-labs/Dia-1.6B", device="cpu")

# Create optimized generate function with INT8 quantization
fast_generate = create_cpu_generate(model, quantize=True)

# Generate audio
audio = fast_generate("[S1] Hello world!")
```

## Optimization Techniques

### 1. `torch.compile` (Easy, 1.5-2x speedup)

The simplest optimization - just enable `use_torch_compile=True`:

```python
audio = model.generate(
    "[S1] Hello world!",
    use_torch_compile=True,  # Enable compilation
)
```

**How it works:** PyTorch compiles the decode step into optimized CUDA kernels, reducing Python overhead and fusing operations.

**Notes:**
- First generation is slow (compilation time)
- Subsequent generations are much faster
- Use `mode="reduce-overhead"` for inference (already set)

### 2. Mixed Precision / FP16 (1.3-2x speedup)

Run computations in half precision for faster math:

```python
import torch

# Manual approach
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    audio = model.generate("[S1] Hello world!")

# Or use the fast_inference module which does this automatically
from dia.fast_inference import create_fast_generate
fast_generate = create_fast_generate(model)
audio = fast_generate("[S1] Hello world!")
```

**Device compatibility:**
- **Ampere+ (RTX 30xx, A100, etc.):** Use `torch.bfloat16` (best)
- **Turing (RTX 20xx):** Use `torch.float16`
- **Older GPUs:** Stick with float32

### 3. Flash Attention (Automatic with PyTorch 2.0+)

The model already uses `F.scaled_dot_product_attention()` which automatically selects the fastest attention implementation:

- **Flash Attention 2:** For Ampere+ GPUs (compute capability ≥ 8.0)
- **Memory-efficient attention:** For older GPUs
- **Math attention:** Fallback

To verify Flash Attention is being used:
```python
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    # This will fail if Flash Attention isn't available
    audio = model.generate("[S1] Test")
```

### 4. CUDA Settings

Enable CUDA optimizations:

```python
import torch

# Enable cuDNN autotuner (finds fastest algorithms)
torch.backends.cudnn.benchmark = True

# Use TensorFloat-32 on Ampere+ (faster matmuls)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set matmul precision
torch.set_float32_matmul_precision('high')  # or 'highest' for accuracy
```

### 5. Reduce CFG Overhead

Classifier-Free Guidance (CFG) doubles the computation. Options:

**A. Lower CFG scale** (slight quality reduction):
```python
audio = model.generate("[S1] Hello", cfg_scale=1.5)  # Default is 3.0
```

**B. Disable CFG for short texts** (text conditioning is usually strong enough):
```python
# For very short, simple texts
audio = model.generate("[S1] Hi", cfg_scale=0.0)
```

**C. Apply CFG only to first N tokens** (not yet implemented, would require code changes).

### 6. Batch Multiple Requests

If you have multiple texts to generate, batch them:

```python
# Instead of sequential generation
for text in texts:
    audio = model.generate(text)

# Consider implementing batched generation
# (requires modifications to support batch_size > 2)
```

## Advanced Optimizations

### Quantization (INT8/INT4)

For maximum speed with some quality tradeoff:

```python
# Using bitsandbytes for 8-bit inference
pip install bitsandbytes

import bitsandbytes as bnb

# Replace linear layers with 8-bit versions
# (requires model modification)
```

### ONNX / TensorRT Export

For production deployment:

```bash
# Export to ONNX
python -c "
import torch
from dia.model import Dia

model = Dia.from_pretrained('nari-labs/Dia-1.6B', device='cuda')
# Export encoder and decoder separately
# ... (complex due to dynamic shapes)
"
```

### Speculative Decoding

For multi-codebook models, speculative decoding can help:
1. Use a smaller draft model to propose tokens
2. Verify with the full model in parallel
3. Accept matching tokens

This is particularly effective for the 9-channel codebook structure.

## Benchmarking

Use the provided benchmark script:

```bash
# Basic benchmark
python example/fast_generation.py --benchmark --level 2

# Compare optimization levels
for level in 1 2 3; do
    echo "Level $level:"
    python example/fast_generation.py --benchmark --level $level
done
```

## Expected Speedups

| Optimization | Speedup | Notes |
|--------------|---------|-------|
| torch.compile | 1.5-2x | After warmup |
| FP16/BF16 | 1.3-2x | Memory-bound ops |
| Flash Attention | 1.2-1.5x | Long sequences |
| cuDNN benchmark | 1.1x | Conv ops |
| All combined | 2-4x | Typical |

## Troubleshooting

### "CUDA out of memory"
- Reduce `max_tokens`
- Use gradient checkpointing (for training)
- Use smaller batch size

### "torch.compile is slow"
- First run includes compilation time
- Use `torch._dynamo.config.cache_size_limit = 64` for repeated calls

### "Flash Attention not available"
- Check CUDA compute capability: `print(torch.cuda.get_device_capability())`
- Need 8.0+ for Flash Attention 2
- Install flash-attn package for better compatibility

### Generation quality degraded with FP16
- Use BF16 instead (better dynamic range)
- Keep final sampling in FP32 (already done)

## CPU-Specific Optimizations

If you don't have a GPU or need to run on CPU, these optimizations are essential.

### 1. Thread Configuration

Properly configure threads for your CPU:

```python
from dia.fast_inference import optimize_for_cpu

# Auto-detect optimal thread count
optimize_for_cpu()

# Or specify manually (use physical cores, not logical)
optimize_for_cpu(num_threads=8)
```

**Guidelines:**
- Use **physical cores** (not hyperthreads)
- For Intel: physical cores = logical cores / 2
- Leave 1-2 cores for system overhead
- Example: 12-core CPU → use 10 threads

### 2. INT8 Dynamic Quantization (2-4x speedup)

The most impactful CPU optimization:

```python
from dia.fast_inference import create_cpu_generate

# Quantization is applied automatically
fast_generate = create_cpu_generate(model, quantize=True)
```

How it works:
- Weights are stored as INT8 (4x smaller)
- Computations use INT8 (faster on modern CPUs)
- Minimal quality loss for TTS

### 3. Intel Extension for PyTorch (IPEX)

For Intel CPUs, IPEX provides additional optimizations:

```bash
pip install intel-extension-for-pytorch
```

```python
from dia.fast_inference import apply_ipex_optimization

# Apply IPEX optimizations
model.model = apply_ipex_optimization(model.model)
```

Benefits:
- Optimized operators for Intel CPUs
- BF16 support on newer Intel CPUs (Sapphire Rapids+)
- Automatic operator fusion

### 4. MKL-DNN Backend

Enabled automatically by `optimize_for_cpu()`, but you can verify:

```python
import torch

# Check if MKL-DNN is available
print(torch.backends.mkldnn.is_available())  # Should be True
print(torch.backends.mkldnn.enabled)          # Should be True
```

### 5. Memory Layout Optimization

For convolution-heavy operations:

```python
# Convert model to channels_last format
model = model.to(memory_format=torch.channels_last)
```

### 6. torch.compile for CPU

PyTorch 2.0+ can compile models for CPU:

```python
# Inductor backend works best for CPU
compiled_model = torch.compile(
    model,
    backend="inductor",
    mode="reduce-overhead"
)
```

### Complete CPU Setup Example

```python
import torch
from dia.model import Dia
from dia.fast_inference import (
    optimize_for_cpu, 
    create_cpu_generate,
    check_cpu_features
)

# 1. Check CPU capabilities
check_cpu_features()

# 2. Configure threading
optimize_for_cpu(num_threads=8)

# 3. Load model on CPU
model = Dia.from_pretrained("nari-labs/Dia-1.6B", device="cpu")

# 4. Create optimized generate function
fast_generate = create_cpu_generate(
    model,
    quantize=True,      # INT8 quantization
    use_compile=True    # torch.compile
)

# 5. Generate
audio = fast_generate("[S1] Hello world!")
```

### CPU Environment Variables

Set these before running for best performance:

```bash
# For Intel CPUs
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

# Disable memory allocator fragmentation
export MALLOC_TRIM_THRESHOLD_=0

# Run inference
python your_script.py
```

## Hardware Recommendations

### GPU Performance

| GPU | Expected Performance | Notes |
|-----|---------------------|-------|
| RTX 4090 | ~0.3-0.5x real-time | Best consumer GPU |
| RTX 3090 | ~0.5-0.8x real-time | Good balance |
| A100 | ~0.2-0.4x real-time | Best for production |
| RTX 3060 | ~1-2x real-time | Budget option |
| RTX 2080 | ~1.5-2.5x real-time | Older but usable |

### CPU Performance

| CPU | Expected Performance | Notes |
|-----|---------------------|-------|
| Intel i9-13900K | ~5-10x real-time | With quantization |
| Intel i7-12700 | ~8-15x real-time | With quantization |
| AMD Ryzen 9 7950X | ~6-12x real-time | With quantization |
| Apple M2 Ultra | ~4-8x real-time | MPS backend recommended |
| Intel Xeon (server) | ~3-8x real-time | Many cores help |

*Real-time factor < 1.0 means faster than real-time*

### CPU vs GPU Comparison

| Aspect | GPU | CPU |
|--------|-----|-----|
| Speed | 5-20x faster | Baseline |
| Memory | Limited by VRAM | Uses system RAM |
| Cost | Expensive | Included |
| Batch size | Larger batches | Single request |
| Quantization benefit | 1.5-2x | 2-4x |

**Recommendation:** Use GPU if available. For CPU-only, quantization is essential.
