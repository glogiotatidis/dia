#!/usr/bin/env python3
"""
Greek Language Training Script for DIA Multilingual TTS

This script trains the DIA model specifically for Greek language.
It can be used for:
- Fine-tuning from a pretrained multilingual checkpoint
- Training from scratch on Greek-only data
- Mixed training with Greek + other languages

Usage:
    # Train on Greek only
    python scripts/train_greek.py --manifest data/el/manifests/train_manifest_el.json
    
    # Fine-tune from pretrained
    python scripts/train_greek.py --manifest data/el/manifests/train_manifest_el.json --pretrained checkpoints/multilang.pt
    
    # Mixed training (Greek + English)
    python scripts/train_greek.py --manifest data/mixed/train_manifest.json --langs el,en
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.multilang_tts_dataset import MultilangTTSDataset, collate_fn
from tools.speaker_encoder import SpeakerEncoder
from dia.model import DiaModel
from dia.config import DiaConfig


def load_config():
    """Load or create default configuration for Greek training."""
    return {
        "model": {
            "encoder_vocab_size": 512,  # Phoneme vocabulary size
            "decoder": {
                "d_model": 512,
            },
            "tgt_vocab_size": 1028,
            "input_dim": 80,  # Mel spectrogram bins
            "diffusion_steps": 8,
        },
        "training": {
            "batch_size": 16,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 1000,
            "max_grad_norm": 1.0,
            "epochs": 50,
            "save_every": 5,
            "eval_every": 1,
            "log_every": 100,
        },
        "data": {
            "sample_rate": 22050,
            "max_audio_len": 15.0,  # seconds
            "min_audio_len": 1.0,   # seconds
        }
    }


class GreekTrainer:
    def __init__(
        self,
        manifest_path: str,
        lang_vocab_path: str,
        output_dir: str,
        pretrained_path: str = None,
        config: dict = None,
        device: str = None,
        use_wandb: bool = False,
    ):
        self.config = config or load_config()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Load language vocabulary
        with open(lang_vocab_path) as f:
            self.lang_vocab = json.load(f)
        
        print(f"🔧 Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")
        
        # Initialize dataset
        print("📚 Loading dataset...")
        self.dataset = MultilangTTSDataset(manifest_path, self.lang_vocab)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False,
        )
        print(f"   Samples: {len(self.dataset)}")
        print(f"   Batches: {len(self.dataloader)}")
        
        # Initialize model
        print("🏗️  Initializing model...")
        self.model = DiaModel(self.config["model"])
        
        if pretrained_path:
            print(f"📥 Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        
        self.model = self.model.to(self.device)
        
        # Initialize speaker encoder
        print("🎤 Loading speaker encoder...")
        self.spk_encoder = SpeakerEncoder(device=self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["lr"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["training"]["lr"],
            epochs=self.config["training"]["epochs"],
            steps_per_epoch=len(self.dataloader),
            pct_start=0.1,
        )
        
        # Initialize wandb if requested
        if self.use_wandb:
            import wandb
            wandb.init(
                project="dia-greek-tts",
                config=self.config,
                name=f"greek-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch["input_ids"] = batch["input_ids"].to(self.device)
            batch["audio"] = batch["audio"].to(self.device)
            batch["lang_token_ids"] = batch["lang_token_ids"].to(self.device)
            
            # Get speaker embeddings
            with torch.no_grad():
                spk_embeds = []
                for path in batch.get("paths", []):
                    try:
                        spk_embeds.append(self.spk_encoder.encode(path))
                    except Exception:
                        # Use zero embedding if speaker encoding fails
                        spk_embeds.append(torch.zeros(192))
                
                if spk_embeds:
                    batch["spk_embed"] = torch.stack(spk_embeds).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["max_grad_norm"]
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config["training"]["log_every"] == 0:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": epoch * len(self.dataloader) + batch_idx,
                })
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"greek_epoch{epoch:03d}_loss{loss:.4f}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Also save as "latest"
        latest_path = self.output_dir / "greek_latest.pt"
        torch.save(self.model.state_dict(), latest_path)
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("🚀 Starting Greek TTS Training")
        print("="*50 + "\n")
        
        best_loss = float("inf")
        
        for epoch in range(self.config["training"]["epochs"]):
            epoch_loss = self.train_epoch(epoch)
            
            print(f"\n📊 Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"   Average Loss: {epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config["training"]["save_every"] == 0:
                self.save_checkpoint(epoch + 1, epoch_loss)
            
            # Track best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = self.output_dir / "greek_best.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"   ⭐ New best model saved!")
            
            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch/loss": epoch_loss,
                    "epoch/best_loss": best_loss,
                    "epoch": epoch + 1,
                })
        
        print("\n" + "="*50)
        print("✅ Training Complete!")
        print(f"   Best Loss: {best_loss:.4f}")
        print(f"   Checkpoints: {self.output_dir}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Train DIA model on Greek language")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to training manifest JSON")
    parser.add_argument("--lang_vocab", type=str, default="configs/lang_vocab.json",
                        help="Path to language vocabulary JSON")
    parser.add_argument("--output_dir", type=str, default="checkpoints/greek",
                        help="Output directory for checkpoints")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model for fine-tuning")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps)")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Update config with command line args
    config = load_config()
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["lr"] = args.lr
    
    # Initialize trainer
    trainer = GreekTrainer(
        manifest_path=args.manifest,
        lang_vocab_path=args.lang_vocab,
        output_dir=args.output_dir,
        pretrained_path=args.pretrained,
        config=config,
        device=args.device,
        use_wandb=args.wandb,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
