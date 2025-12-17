#!/usr/bin/env python3
"""
Greek TTS Inference Script

Test your trained Greek model by generating audio from text.

Usage:
    # Using a trained checkpoint
    python scripts/infer_greek.py --checkpoint checkpoints/greek/greek_best.pt --text "Γεια σου κόσμε"
    
    # Using the pretrained Dia model (for comparison)
    python scripts/infer_greek.py --pretrained --text "Hello world"
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import soundfile as sf
import torch

from dia.model import Dia
from dia.config import DiaConfig, DataConfig, EncoderConfig, DecoderConfig, ModelConfig, TrainingConfig
from dia.layers import DiaModel


def create_default_config():
    """Create default config matching training."""
    encoder_config = EncoderConfig(
        n_layer=12,
        n_embd=768,
        n_hidden=3072,
        n_head=12,
        head_dim=64,
    )
    decoder_config = DecoderConfig(
        n_layer=12,
        n_embd=768,
        n_hidden=3072,
        gqa_query_heads=12,
        kv_heads=4,
        gqa_head_dim=64,
        cross_query_heads=12,
        cross_head_dim=64,
    )
    model_config = ModelConfig(
        encoder=encoder_config,
        decoder=decoder_config,
        src_vocab_size=256,
        tgt_vocab_size=1028,
    )
    training_config = TrainingConfig(dtype="float32")
    data_config = DataConfig(
        text_length=512,
        audio_length=3072,
    )
    
    return DiaConfig(
        model=model_config,
        training=training_config,
        data=data_config,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate audio from Greek text")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (.pt file)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (optional, will use default if not provided)")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use pretrained Dia model from HuggingFace")
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize (Greek or English)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio file path")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--temperature", type=float, default=1.3,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling")
    parser.add_argument("--cfg_scale", type=float, default=3.0,
                        help="Classifier-free guidance scale")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"🔧 Device: {device}")
    
    if args.pretrained:
        # Use pretrained model
        print("📥 Loading pretrained Dia model...")
        dia = Dia.from_pretrained("nari-labs/Dia-1.6B", device=device)
    else:
        if not args.checkpoint:
            print("❌ Error: Either --checkpoint or --pretrained must be specified")
            sys.exit(1)
        
        # Load trained checkpoint
        print(f"📥 Loading checkpoint: {args.checkpoint}")
        
        # Load config
        if args.config:
            config = DiaConfig.load(args.config)
        else:
            # Try to find config next to checkpoint
            checkpoint_dir = Path(args.checkpoint).parent
            config_path = checkpoint_dir / "config.json"
            if config_path.exists():
                config = DiaConfig.load(str(config_path))
                print(f"   Using config from {config_path}")
            else:
                print("   Using default config")
                config = create_default_config()
        
        # Create Dia instance
        dia = Dia(config, device=device)
        
        # Load weights
        state_dict = torch.load(args.checkpoint, map_location=device)
        dia.model.load_state_dict(state_dict, strict=False)
        dia.model.to(device)
        dia.model.eval()
        
        # Load DAC model
        dia._load_dac_model()
    
    # Format text with speaker marker if not present
    text = args.text
    if "[S1]" not in text and "[S2]" not in text:
        text = f"[S1] {text}"
    
    print(f"📝 Input text: {text}")
    print(f"🎵 Generating audio...")
    
    # Generate
    audio = dia.generate(
        text=text,
        temperature=args.temperature,
        top_p=args.top_p,
        cfg_scale=args.cfg_scale,
    )
    
    # Save
    sf.write(args.output, audio, 44100)
    print(f"✅ Audio saved to: {args.output}")
    print(f"   Duration: {len(audio) / 44100:.2f} seconds")


if __name__ == "__main__":
    main()
