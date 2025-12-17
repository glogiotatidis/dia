#!/usr/bin/env python3
"""
Greek Language Training Script for DIA TTS

This script trains the DIA model for Greek language using the proper
architecture with DAC audio tokenization.

Usage:
    python scripts/train_greek.py --manifest data/el/manifests/train_manifest_el.json
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import dac
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.multilang_tts_dataset import MultilangTTSDataset, collate_fn
from dia.model import Dia
from dia.config import DiaConfig, DataConfig, EncoderConfig, DecoderConfig, ModelConfig, TrainingConfig
from dia.audio import audio_to_codebook, build_delay_indices, apply_audio_delay
from dia.layers import DiaModel


def create_default_config():
    """Create a default DIA config for training."""
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
        src_vocab_size=256,  # Byte vocabulary
        tgt_vocab_size=1028,  # DAC codes + special tokens
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


class GreekTrainer:
    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        pretrained_path: str = None,
        config: DiaConfig = None,
        device: str = None,
        batch_size: int = 4,
        lr: float = 1e-4,
        epochs: int = 50,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        
        print(f"🔧 Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")
        
        # Load or create config
        if config is None:
            if pretrained_path:
                # Try to load config from pretrained
                config_path = Path(pretrained_path).parent / "config.json"
                if config_path.exists():
                    config = DiaConfig.load(str(config_path))
            if config is None:
                print("📝 Using default config")
                config = create_default_config()
        
        self.config = config
        
        # Initialize dataset with empty lang_vocab (we use byte encoding)
        print("📚 Loading dataset...")
        self.dataset = MultilangTTSDataset(manifest_path, lang_vocab={})
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True if self.device == "cuda" else False,
        )
        print(f"   Samples: {len(self.dataset)}")
        print(f"   Batches: {len(self.dataloader)}")
        
        # Initialize model
        print("🏗️  Initializing model...")
        self.model = DiaModel(config)
        
        if pretrained_path:
            print(f"📥 Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model = self.model.to(self.device)
        
        # Load DAC model for audio tokenization
        print("🎵 Loading DAC model...")
        try:
            dac_model_path = dac.utils.download()
            self.dac_model = dac.DAC.load(dac_model_path).to(self.device)
            self.dac_model.eval()
            print("✅ DAC model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load DAC model: {e}")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01,
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs * len(self.dataloader),
            eta_min=lr * 0.1,
        )
        
        # Loss function - cross entropy for token prediction
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.data.audio_pad_value)
    
    def encode_audio_to_codes(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Encode waveforms to DAC codes.
        
        Args:
            waveforms: (B, samples) audio waveforms at 44.1kHz
            
        Returns:
            codes: (B, T, C) audio codes where C=9 channels
        """
        with torch.no_grad():
            # DAC expects (B, 1, samples)
            audio_input = waveforms.unsqueeze(1).to(self.device)
            
            # Encode
            audio_data = self.dac_model.preprocess(audio_input, 44100)
            _, codes, _, _, _ = self.dac_model.encode(audio_data, n_quantizers=None)
            # codes shape: (B, C, T)
            
            # Transpose to (B, T, C)
            codes = codes.transpose(1, 2)
            
            # Apply delay pattern
            B, T, C = codes.shape
            t_idx, indices = build_delay_indices(
                B=B, T=T, C=C,
                delay_pattern=self.config.data.delay_pattern
            )
            codes = apply_audio_delay(
                codes,
                pad_value=self.config.data.audio_pad_value,
                bos_value=self.config.data.audio_bos_value,
                precomp=(t_idx, indices)
            )
            
        return codes
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Get text tokens and waveforms
                text_tokens = batch["text_tokens"].to(self.device)  # (B, S)
                waveforms = batch["waveforms"]  # (B, samples) - keep on CPU for DAC
                
                # Encode audio to DAC codes
                audio_codes = self.encode_audio_to_codes(waveforms)  # (B, T, C)
                
                # Create positions
                B, S = text_tokens.shape
                _, T, C = audio_codes.shape
                
                src_positions = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)
                tgt_positions = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
                
                # Create masks
                src_padding_mask = (text_tokens != 0)  # (B, S)
                
                # Causal mask for decoder self-attention
                causal_mask = torch.tril(torch.ones(T, T, device=self.device)).bool()
                dec_self_attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
                
                # Cross-attention mask
                dec_cross_attn_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
                
                # Encoder self-attention mask
                enc_self_attn_mask = src_padding_mask.unsqueeze(1).unsqueeze(2) & src_padding_mask.unsqueeze(1).unsqueeze(3)
                
                # Forward pass - shift targets for teacher forcing
                # Input: all but last token, Target: all but first token
                tgt_input = audio_codes[:, :-1, :]  # (B, T-1, C)
                tgt_target = audio_codes[:, 1:, :]  # (B, T-1, C)
                
                tgt_positions = tgt_positions[:, :-1]
                dec_self_attn_mask = dec_self_attn_mask[:, :, :-1, :-1]
                
                # Forward
                self.optimizer.zero_grad()
                
                logits = self.model(
                    src_BxS=text_tokens,
                    tgt_BxTxC=tgt_input,
                    src_positions=src_positions,
                    tgt_positions=tgt_positions,
                    enc_self_attn_mask=enc_self_attn_mask,
                    dec_self_attn_mask=dec_self_attn_mask,
                    dec_cross_attn_mask=dec_cross_attn_mask,
                    enable_dropout=True,
                )
                # logits: (B, T-1, C, V)
                
                # Compute loss across all channels
                B, T_out, C, V = logits.shape
                logits_flat = logits.reshape(-1, V)  # (B*T*C, V)
                targets_flat = tgt_target.reshape(-1).long()  # (B*T*C,)
                
                loss = self.criterion(logits_flat, targets_flat)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
            except Exception as e:
                print(f"\n⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"greek_epoch{epoch:03d}_loss{loss:.4f}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Also save as "latest"
        latest_path = self.output_dir / "greek_latest.pt"
        torch.save(self.model.state_dict(), latest_path)
        
        # Save config
        config_path = self.output_dir / "config.json"
        self.config.save(str(config_path))
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("🚀 Starting Greek TTS Training")
        print("="*50 + "\n")
        
        best_loss = float("inf")
        
        for epoch in range(self.epochs):
            epoch_loss = self.train_epoch(epoch)
            
            print(f"\n📊 Epoch {epoch+1}/{self.epochs}")
            print(f"   Average Loss: {epoch_loss:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, epoch_loss)
            
            # Track best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = self.output_dir / "greek_best.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"   ⭐ New best model saved!")
        
        print("\n" + "="*50)
        print("✅ Training Complete!")
        print(f"   Best Loss: {best_loss:.4f}")
        print(f"   Checkpoints: {self.output_dir}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Train DIA model on Greek language")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to training manifest JSON")
    parser.add_argument("--output_dir", type=str, default="checkpoints/greek",
                        help="Output directory for checkpoints")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model for fine-tuning")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (reduced for memory)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = GreekTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        pretrained_path=args.pretrained,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
