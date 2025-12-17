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
        from_hf: bool = False,
        config: DiaConfig = None,
        device: str = None,
        batch_size: int = 1,
        grad_accum: int = 8,
        lr: float = 1e-5,
        epochs: int = 50,
        max_audio_len: float = 10.0,
        freeze_encoder: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.lr = lr
        self.max_audio_len = max_audio_len
        
        print(f"🔧 Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")
        print(f"📊 Batch size: {batch_size} x {grad_accum} grad accum = {batch_size * grad_accum} effective")
        
        # Load pretrained from HuggingFace (recommended)
        if from_hf:
            print("📥 Loading pretrained Dia-1.6B from HuggingFace...")
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id="nari-labs/Dia-1.6B", filename="config.json")
            checkpoint_path = hf_hub_download(repo_id="nari-labs/Dia-1.6B", filename="dia-v0_1.pth")
            config = DiaConfig.load(config_path)
            pretrained_path = checkpoint_path
            print(f"   ✅ Downloaded pretrained model")
        elif config is None:
            if pretrained_path:
                # Try to load config from pretrained
                config_path = Path(pretrained_path).parent / "config.json"
                if config_path.exists():
                    config = DiaConfig.load(str(config_path))
            if config is None:
                print("⚠️  WARNING: Training from scratch without pretrained weights!")
                print("   This will likely produce poor results. Use --from_hf for better results.")
                config = create_default_config()
        
        self.config = config
        
        # Initialize dataset with empty lang_vocab (we use byte encoding)
        print("📚 Loading dataset...")
        self.dataset = MultilangTTSDataset(manifest_path, lang_vocab={}, max_audio_len=max_audio_len)
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
            print(f"📥 Loading pretrained weights...")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
            print(f"   ✅ Loaded weights from {pretrained_path}")
        
        # Optionally freeze encoder to save memory
        if freeze_encoder:
            print("🧊 Freezing encoder (training decoder only)...")
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        else:
            print("🔥 Training full model (encoder + decoder)")
        
        self.model = self.model.to(self.device)
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"   Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M parameters")
        
        # Load DAC model for audio tokenization
        print("🎵 Loading DAC model...")
        try:
            dac_model_path = dac.utils.download()
            self.dac_model = dac.DAC.load(dac_model_path).to(self.device)
            self.dac_model.eval()
            print("✅ DAC model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load DAC model: {e}")
        
        # Initialize optimizer with epsilon for numerical stability
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01,
            eps=1e-8,  # Numerical stability
        )
        
        # Learning rate scheduler with warmup
        total_steps = epochs * len(self.dataloader)
        warmup_steps = min(500, total_steps // 10)  # 10% warmup or 500 steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps  # Linear warmup
            return 0.5 * (1 + torch.cos(torch.tensor((step - warmup_steps) / (total_steps - warmup_steps) * 3.14159)).item())
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # GradScaler for mixed precision (critical for preventing NaN!)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda"))
        
        # Loss function - cross entropy for token prediction
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.data.audio_pad_value)
        
        # NaN tracking
        self.nan_count = 0
        self.max_nan_batches = 10  # Stop training if too many NaN batches
    
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
        """Train for one epoch with gradient accumulation and NaN protection."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
        self.optimizer.zero_grad()
        
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
                
                # Forward with mixed precision and GradScaler
                with torch.cuda.amp.autocast(enabled=self.device == "cuda"):
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
                    loss = loss / self.grad_accum  # Scale for accumulation
                
                # Check for NaN loss BEFORE backward
                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count += 1
                    print(f"\n⚠️ NaN/Inf loss detected in batch {batch_idx} (count: {self.nan_count}/{self.max_nan_batches})")
                    
                    if self.nan_count >= self.max_nan_batches:
                        print("❌ Too many NaN batches! Stopping training.")
                        print("   Try: lower learning rate, check data quality, or reduce batch size")
                        raise RuntimeError("Training diverged with too many NaN losses")
                    
                    # Skip this batch and reset gradients
                    self.optimizer.zero_grad()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    continue
                
                # Backward with scaler (handles FP16 scaling)
                self.scaler.scale(loss).backward()
                
                # Update weights every grad_accum steps
                if (batch_idx + 1) % self.grad_accum == 0:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Check for NaN gradients
                    has_nan_grad = False
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        self.nan_count += 1
                        print(f"\n⚠️ NaN gradients detected (count: {self.nan_count}/{self.max_nan_batches})")
                        self.optimizer.zero_grad()
                        
                        if self.nan_count >= self.max_nan_batches:
                            print("❌ Too many NaN gradient batches! Stopping training.")
                            raise RuntimeError("Training diverged with NaN gradients")
                        continue
                    
                    # Gradient clipping (important for stability!)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Step with scaler (skips update if gradients are inf)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                
                total_loss += loss.item() * self.grad_accum  # Unscale for logging
                num_batches += 1
                
                # Clear cache periodically
                if batch_idx % 10 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()
                
                pbar.set_postfix({
                    "loss": f"{loss.item() * self.grad_accum:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
            except torch.cuda.OutOfMemoryError:
                print(f"\n⚠️ OOM in batch {batch_idx}, skipping...")
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                continue
            except RuntimeError as e:
                if "NaN" in str(e) or "diverged" in str(e):
                    raise  # Re-raise NaN errors
                print(f"\n⚠️ Error in batch {batch_idx}: {e}")
                continue
            except Exception as e:
                print(f"\n⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Check if epoch average loss is NaN
        if num_batches == 0 or avg_loss != avg_loss:  # NaN check
            print(f"⚠️ Epoch {epoch+1} had no valid batches or NaN average loss!")
            return float('nan')
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint (only if loss is valid)."""
        # Don't save checkpoints with NaN loss
        if loss != loss or loss == float('inf'):  # NaN check
            print(f"⚠️ Skipping checkpoint save - loss is NaN/Inf")
            return
        
        # Use descriptive filename
        loss_str = f"{loss:.4f}".replace(".", "p")  # e.g., 2p3456
        checkpoint_path = self.output_dir / f"greek_epoch{epoch:03d}_loss{loss_str}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Also save as "latest"
        latest_path = self.output_dir / "greek_latest.pt"
        torch.save(self.model.state_dict(), latest_path)
        
        # Save config
        config_path = self.output_dir / "config.json"
        self.config.save(str(config_path))
    
    def train(self):
        """Main training loop with NaN protection."""
        print("\n" + "="*50)
        print("🚀 Starting Greek TTS Training")
        print("="*50 + "\n")
        
        best_loss = float("inf")
        nan_epochs = 0
        max_nan_epochs = 3  # Stop if 3 consecutive epochs have NaN
        
        for epoch in range(self.epochs):
            try:
                epoch_loss = self.train_epoch(epoch)
            except RuntimeError as e:
                if "diverged" in str(e) or "NaN" in str(e):
                    print(f"\n❌ Training stopped due to: {e}")
                    break
                raise
            
            print(f"\n📊 Epoch {epoch+1}/{self.epochs}")
            
            # Check for NaN epoch loss
            if epoch_loss != epoch_loss:  # NaN check
                nan_epochs += 1
                print(f"   ⚠️ Epoch loss is NaN! (consecutive: {nan_epochs}/{max_nan_epochs})")
                
                if nan_epochs >= max_nan_epochs:
                    print(f"\n❌ Training stopped: {max_nan_epochs} consecutive NaN epochs")
                    print("   Suggestions:")
                    print("   - Lower learning rate (try --lr 1e-6)")
                    print("   - Check dataset for corrupt audio files")
                    print("   - Reduce max_audio_len")
                    break
                continue
            else:
                nan_epochs = 0  # Reset counter on valid epoch
            
            print(f"   Average Loss: {epoch_loss:.4f}")
            
            # Save checkpoint every 5 epochs (only valid losses)
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, epoch_loss)
            
            # Track best model (only if loss improved)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = self.output_dir / "greek_best.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"   ⭐ New best model saved!")
        
        print("\n" + "="*50)
        if best_loss == float("inf"):
            print("❌ Training Failed - No valid checkpoints saved")
            print("   All epochs had NaN loss. Check your data and hyperparameters.")
        else:
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
                        help="Path to pretrained model checkpoint (.pt/.pth)")
    parser.add_argument("--from_hf", action="store_true",
                        help="Start from pretrained Dia-1.6B from HuggingFace (recommended)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (use 1 for 16GB GPU)")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--max_audio_len", type=float, default=10.0,
                        help="Max audio length in seconds (reduce for memory)")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder, only train decoder (saves memory)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = GreekTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        pretrained_path=args.pretrained,
        from_hf=args.from_hf,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        max_audio_len=args.max_audio_len,
        freeze_encoder=args.freeze_encoder,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
