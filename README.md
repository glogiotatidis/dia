# DIA-Multilingual (StyleTTS2-based)

This is a fork of the original DIA model extended for **multilingual TTS**, with support for 30+ languages (same as ElevenLabs). Built on top of StyleTTS2 with language token injection, espeak-ng phonemization, and support for reference audio-based style transfer.

---

## 🧠 Supported Languages

Supports over 30 languages via `<lang>` token injection, including:

`en`, `es`, `de`, `fr`, `it`, `pt`, `pl`, `ro`, `nl`, `tr`, `sv`, `cs`, `el`, `hu`, `fi`, `da`, `sk`, `bg`, `hi`, `ar`, `zh`, `ja`, `ko`, ...

---

## 🚀 Quickstart (RunPod)

**Build your container:**
```bash
docker build -t dia-multilang -f docker/Dockerfile .
```

**Launch training (inside container):**
```bash
bash docker/launch.sh
```

This will:
- Load `train_manifest.json` and `lang_vocab.json`
- Start training from scratch using espeak-based phoneme inputs
- Save checkpoints in `/workspace/checkpoints`

---

## 🧾 File Structure

```bash
├── dataset/
│   └── multilang_tts_dataset.py     # Dataset + phonemizer + collate_fn
├── scripts/
│   ├── train_dia.py                 # Main training loop
│   ├── validate.py                  # Eval script (loss only)
│   └── infer_dia.py                 # Generate audio from text
├── docker/
│   ├── Dockerfile                   # GPU-enabled training container
│   └── launch.sh                    # Entrypoint script
├── lang_vocab.json                  # Maps <lang> → token_id
├── train_manifest.json              # Manifest (audio, text, lang)
```

---

## 🎙️ Inference

Generate audio from text using:

```bash
python3 scripts/infer_dia.py \
  --model_path checkpoints/epoch49.pt \
  --lang_vocab lang_vocab.json \
  --text "Ciao, come stai?" \
  --lang it \
  --output_dir samples/
```

To use reference audio (zero-shot style cloning):

```bash
  --reference_wav samples/italian_female.wav
```

---

## ⚡ Fast Inference

Optimized inference for both GPU and CPU. See [docs/INFERENCE_OPTIMIZATION.md](docs/INFERENCE_OPTIMIZATION.md) for the full guide.

### GPU (2-4x faster)

```bash
python example/fast_generation.py --text "[S1] Hello world" --level 3
```

Or programmatically:

```python
from dia.model import Dia
from dia.fast_inference import create_fast_generate

model = Dia.from_pretrained("nari-labs/Dia-1.6B", device="cuda")
fast_generate = create_fast_generate(model)
audio = fast_generate("[S1] Hello world!")
```

### CPU with Quantization (2-4x faster than baseline CPU)

```bash
python example/fast_generation.py --cpu --quantize --text "[S1] Hello world"
```

Or programmatically:

```python
from dia.model import Dia
from dia.fast_inference import optimize_for_cpu, create_cpu_generate

optimize_for_cpu(num_threads=8)
model = Dia.from_pretrained("nari-labs/Dia-1.6B", device="cpu")
fast_generate = create_cpu_generate(model, quantize=True)
audio = fast_generate("[S1] Hello world!")
```

### Benchmark

```bash
# GPU benchmark
python example/fast_generation.py --benchmark --level 3

# CPU benchmark
python example/fast_generation.py --cpu --quantize --benchmark
```

### Optimization Summary

| Device | Optimization | Speedup |
|--------|--------------|---------|
| GPU | torch.compile + FP16 | 2-3x |
| GPU | + INT8 quantization | 3-4x |
| CPU | INT8 quantization | 2-4x |
| CPU | + torch.compile | 3-5x |

---

## 💡 Notes

- Uses espeak-ng to phonemize all input text (per-language IPA)
- Pretrained `xlm-roberta-base` recommended for phoneme encodings
- Output speech is high-fidelity and respects cross-language style transfer

---

## 🧠 Credits
Built on top of:
- [DIA](https://github.com/nari-labs/dia)
- [StyleTTS2](https://github.com/yl4579/StyleTTS2)