#!/usr/bin/env python3
"""
Greek TTS Training Script - Optimized for Apple Silicon (M1/M2/M3)

This script is optimized for training on Mac with MPS (Metal Performance Shaders).
Includes memory optimizations and appropriate batch sizes for unified memory.

Usage:
    python scripts/train_greek_mac.py --manifest data/el/manifests/train_manifest_el.json
"""

import argparse
import json
import os
import gc
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check for MPS availability
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🍎 Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("🖥️  Using NVIDIA GPU (CUDA)")
else:
    DEVICE = "cpu"
    print("⚠️  Using CPU (slow)")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.multilang_tts_dataset import MultilangTTSDataset, collate_fn
from dia.model import DiaModel


def get_mac_optimized_config():
    """Configuration optimized for M3 Pro with 36GB RAM."""
    return {
        "model": {
            "encoder_vocab_size": 512,
            "decoder": {"d_model": 512},
            "tgt_vocab_size": 1028,
            "input_dim": 80,
            "diffusion_steps": 8,
        },
        "training": {
            # Smaller batch size for unified memory
            "batch_size": 8,  # Reduce to 4 if OOM
            "lr": 1e-4,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "epochs": 50,
            "save_every": 5,
            "log_every": 50,
            # Mac-specific optimizations
            "gradient_accumulation_steps": 2,  # Effective batch = 16
            "mixed_precision": False,  # MPS doesn't fully support AMP yet
            "num_workers": 0,  # DataLoader workers can cause issues on Mac
        },
        "data": {
            "sample_rate": 22050,
            "max_audio_len": 10.0,  # Shorter clips use less memory
            "min_audio_len": 1.0,
        }
    }


class MacOptimizedTrainer:
    def __init__(
        self,
        manifest_path: str,
        lang_vocab_path: str,
        output_dir: str,
        config: dict = None,
    ):
        self.config = config or get_mac_optimized_config()
        self.device = DEVICE
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load language vocabulary
        with open(lang_vocab_path) as f:
            self.lang_vocab = json.load(f)
        
        print(f"\n🔧 Configuration:")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.config['training']['batch_size']}")
        print(f"   Gradient accumulation: {self.config['training']['gradient_accumulation_steps']}")
        print(f"   Effective batch: {self.config['training']['batch_size'] * self.config['training']['gradient_accumulation_steps']}")
        
        # Initialize dataset
        print("\n📚 Loading dataset...")
        self.dataset = MultilangTTSDataset(manifest_path, self.lang_vocab)
        
        # Use num_workers=0 on Mac to avoid multiprocessing issues
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config["training"]["num_workers"],
            pin_memory=False,  # Not needed for MPS
        )
        print(f"   Samples: {len(self.dataset)}")
        print(f"   Batches per epoch: {len(self.dataloader)}")
        
        # Estimate training time
        self._estimate_time()
        
        # Initialize model
        print("\n🏗️  Initializing model...")
        self.model = DiaModel(self.config["model"])
        self.model = self.model.to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {n_params/1e6:.1f}M")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["lr"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        
        # Simple LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["training"]["epochs"],
        )
        
        self.grad_accum_steps = self.config["training"]["gradient_accumulation_steps"]
    
    def _estimate_time(self):
        """Estimate total training time."""
        n_batches = len(self.dataloader)
        n_epochs = self.config["training"]["epochs"]
        
        # Rough estimate: ~0.5-1 sec per batch on M3 Pro
        sec_per_batch = 0.75
        total_batches = n_batches * n_epochs
        total_hours = (total_batches * sec_per_batch) / 3600
        
        print(f"\n⏱️  Estimated training time:")
        print(f"   Per epoch: ~{n_batches * sec_per_batch / 60:.0f} minutes")
        print(f"   Total ({n_epochs} epochs): ~{total_hours:.1f} hours")
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch with gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch["input_ids"] = batch["input_ids"].to(self.device)
            batch["audio"] = batch["audio"].to(self.device)
            batch["lang_token_ids"] = batch["lang_token_ids"].to(self.device)
            
            # Forward pass
            loss = self.model(batch)
            loss = loss / self.grad_accum_steps  # Scale for accumulation
            
            # Backward pass
            loss.backward()
            
            # Update weights every grad_accum_steps
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["max_grad_norm"]
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Clear MPS cache periodically
                if self.device == "mps" and batch_idx % 100 == 0:
                    torch.mps.empty_cache()
            
            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * self.grad_accum_steps:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        self.scheduler.step()
        
        # Clear memory after epoch
        if self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save checkpoint."""
        checkpoint_path = self.output_dir / f"greek_epoch{epoch:03d}.pt"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"💾 Saved: {checkpoint_path}")
        
        # Keep latest
        latest_path = self.output_dir / "greek_latest.pt"
        torch.save(self.model.state_dict(), latest_path)
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*50)
        print("🚀 Starting Training on Apple Silicon")
        print("="*50)
        
        best_loss = float("inf")
        start_time = datetime.now()
        
        for epoch in range(self.config["training"]["epochs"]):
            epoch_start = datetime.now()
            epoch_loss = self.train_epoch(epoch)
            epoch_time = (datetime.now() - epoch_start).total_seconds() / 60
            
            print(f"\n📊 Epoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"   Loss: {epoch_loss:.4f}")
            print(f"   Time: {epoch_time:.1f} min")
            
            # Save checkpoint
            if (epoch + 1) % self.config["training"]["save_every"] == 0:
                self.save_checkpoint(epoch + 1, epoch_loss)
            
            # Track best
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = self.output_dir / "greek_best.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"   ⭐ New best!")
        
        total_time = (datetime.now() - start_time).total_seconds() / 3600
        
        print("\n" + "="*50)
        print("✅ Training Complete!")
        print(f"   Best loss: {best_loss:.4f}")
        print(f"   Total time: {total_time:.1f} hours")
        print(f"   Checkpoints: {self.output_dir}")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Train Greek TTS on Mac")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--lang_vocab", type=str, default="configs/lang_vocab.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/greek")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    
    config = get_mac_optimized_config()
    config["training"]["epochs"] = args.epochs
    config["training"]["batch_size"] = args.batch_size
    
    trainer = MacOptimizedTrainer(
        manifest_path=args.manifest,
        lang_vocab_path=args.lang_vocab,
        output_dir=args.output_dir,
        config=config,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
