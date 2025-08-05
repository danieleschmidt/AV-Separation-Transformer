#!/usr/bin/env python3
"""
Training Script for AV-Separation-Transformer
Supports distributed training, mixed precision, and comprehensive evaluation
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from src.av_separation import AVSeparator, SeparatorConfig
from src.av_separation.models import AVSeparationTransformer
from src.av_separation.utils.losses import CombinedLoss
from src.av_separation.utils.metrics import compute_all_metrics


def setup_logging(log_dir: Path, rank: int = 0):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_level = logging.INFO if rank == 0 else logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / f'train_rank_{rank}.log'),
            logging.StreamHandler() if rank == 0 else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int):
    """Setup distributed training"""
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )
        
        torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


class AudioVideoDataset(torch.utils.data.Dataset):
    """
    Placeholder dataset for audio-video separation training
    In practice, this would load real audio-video data
    """
    
    def __init__(self, data_dir: str, config: SeparatorConfig, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        
        # In practice, this would scan the data directory for files
        # For demonstration, we'll create synthetic data
        self.num_samples = 1000 if split == 'train' else 100
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate synthetic data for demonstration
        # In practice, load real audio-video pairs
        
        # Audio: [time_frames, n_mels]
        audio_len = int(self.config.audio.chunk_duration * 
                       self.config.audio.sample_rate / self.config.audio.hop_length)
        mixture_audio = torch.randn(audio_len, self.config.audio.n_mels)
        
        # Video: [time_frames, channels, height, width]
        video_len = int(self.config.audio.chunk_duration * self.config.video.fps)
        video_frames = torch.randn(video_len, 3, *self.config.video.image_size)
        
        # Target separated audio: [num_speakers, time_frames, n_mels]
        num_speakers = torch.randint(2, self.config.model.max_speakers + 1, (1,)).item()
        target_audio = torch.randn(num_speakers, audio_len, self.config.audio.n_mels)
        
        # Audio waveforms for loss computation: [num_speakers, waveform_length]
        waveform_len = int(self.config.audio.chunk_duration * self.config.audio.sample_rate)
        target_waveforms = torch.randn(num_speakers, waveform_len)
        
        return {
            'mixture_audio': mixture_audio,
            'video_frames': video_frames,
            'target_audio': target_audio,
            'target_waveforms': target_waveforms,
            'num_speakers': num_speakers
        }


def create_data_loaders(config: SeparatorConfig, args) -> Dict[str, DataLoader]:
    """Create training and validation data loaders"""
    
    train_dataset = AudioVideoDataset(args.data_dir, config, 'train')
    val_dataset = AudioVideoDataset(args.data_dir, config, 'val')
    
    # Distributed sampling
    train_sampler = None
    val_sampler = None
    
    if args.world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return {'train': train_loader, 'val': val_loader}


def create_model_and_optimizer(config: SeparatorConfig, args) -> tuple:
    """Create model, optimizer, and scheduler"""
    
    # Create model
    model = AVSeparationTransformer(config)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Move to GPU
    device = torch.device(f'cuda:{args.rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Distributed training
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.rank])
    
    # Optimizer
    if config.training.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
    
    # Scheduler
    if config.training.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs
        )
    elif config.training.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        scheduler = None
    
    return model, optimizer, scheduler, device


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: SeparatorConfig,
    logger: logging.Logger,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Train for one epoch"""
    
    model.train()
    total_loss = 0.0
    total_si_snr = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move data to device
        mixture_audio = batch['mixture_audio'].to(device)
        video_frames = batch['video_frames'].to(device)
        target_waveforms = batch['target_waveforms'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=config.training.mixed_precision):
            outputs = model(mixture_audio, video_frames)
            
            # Extract predictions
            pred_waveforms = outputs['separated_waveforms']
            
            # Compute loss
            loss_dict = criterion(pred_waveforms, target_waveforms)
            loss = loss_dict['total']
        
        # Backward pass
        if config.training.mixed_precision:
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.training.gradient_clip_val > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.training.gradient_clip_val
                )
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            if config.training.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.gradient_clip_val
                )
            
            optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        if 'si_snr' in loss_dict:
            total_si_snr += (-loss_dict['si_snr'].item())  # SI-SNR loss is negative
        
        num_batches += 1
        
        # Log progress
        if batch_idx % 50 == 0:
            logger.info(
                f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                f'Loss: {loss.item():.4f}'
            )
            
            if writer:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_si_snr = total_si_snr / num_batches
    
    return {
        'loss': avg_loss,
        'si_snr': avg_si_snr
    }


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: SeparatorConfig,
    logger: logging.Logger
) -> Dict[str, float]:
    """Validate for one epoch"""
    
    model.eval()
    total_loss = 0.0
    total_si_snr = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            mixture_audio = batch['mixture_audio'].to(device)
            video_frames = batch['video_frames'].to(device)
            target_waveforms = batch['target_waveforms'].to(device)
            
            # Forward pass
            outputs = model(mixture_audio, video_frames)
            pred_waveforms = outputs['separated_waveforms']
            
            # Compute loss
            loss_dict = criterion(pred_waveforms, target_waveforms)
            loss = loss_dict['total']
            
            # Accumulate metrics
            total_loss += loss.item()
            if 'si_snr' in loss_dict:
                total_si_snr += (-loss_dict['si_snr'].item())
            
            num_batches += 1
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_si_snr = total_si_snr / num_batches
    
    logger.info(f'Validation - Loss: {avg_loss:.4f}, SI-SNR: {avg_si_snr:.2f} dB')
    
    return {
        'loss': avg_loss,
        'si_snr': avg_si_snr
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    config: SeparatorConfig,
    checkpoint_dir: Path,
    is_best: bool = False
):
    """Save model checkpoint"""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config
    }
    
    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch:03d}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"Saved best model at epoch {epoch}")


def main(args):
    """Main training function"""
    
    # Setup distributed training
    setup_distributed(args.rank, args.world_size)
    
    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir, args.rank)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = SeparatorConfig.from_dict(config_dict)
    else:
        config = SeparatorConfig()
    
    # Override config with command line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    
    logger.info(f"Training configuration: {config.to_dict()}")
    
    # Create data loaders
    data_loaders = create_data_loaders(config, args)
    
    # Create model, optimizer, and scheduler
    model, optimizer, scheduler, device = create_model_and_optimizer(config, args)
    
    # Create loss function
    criterion = CombinedLoss(
        si_snr_weight=1.0,
        stft_weight=0.5,
        perceptual_weight=0.1,
        use_pit=True
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.training.mixed_precision)
    
    # TensorBoard writer (only on rank 0)
    writer = None
    if args.rank == 0:
        writer = SummaryWriter(log_dir / 'tensorboard')
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    start_epoch = 0
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, config.training.num_epochs):
        # Set epoch for distributed sampler
        if args.world_size > 1:
            data_loaders['train'].sampler.set_epoch(epoch)
        
        # Training
        train_metrics = train_epoch(
            model, data_loaders['train'], criterion, optimizer, scaler,
            device, epoch, config, logger, writer
        )
        
        # Validation
        if epoch % config.training.val_every_n_epochs == 0:
            val_metrics = validate_epoch(
                model, data_loaders['val'], criterion,
                device, epoch, config, logger
            )
            
            # Log to TensorBoard
            if writer:
                writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                writer.add_scalar('Train/SI-SNR', train_metrics['si_snr'], epoch)
                writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                writer.add_scalar('Val/SI-SNR', val_metrics['si_snr'], epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
            
            if args.rank == 0 and epoch % config.training.save_every_n_epochs == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_metrics['loss'],
                    config, checkpoint_dir, is_best
                )
        
        # Step scheduler
        if scheduler:
            scheduler.step()
        
        logger.info(f"Epoch {epoch} completed - Train Loss: {train_metrics['loss']:.4f}")
    
    # Final checkpoint
    if args.rank == 0:
        save_checkpoint(
            model, optimizer, scheduler, config.training.num_epochs - 1,
            train_metrics['loss'], config, checkpoint_dir, False
        )
    
    # Cleanup
    if writer:
        writer.close()
    
    cleanup_distributed()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AV-Separation-Transformer')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint to resume from')
    
    # Logging arguments
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for logs and checkpoints')
    
    # Distributed training arguments
    parser.add_argument('--world-size', type=int, default=1,
                       help='Number of distributed processes')
    parser.add_argument('--rank', type=int, default=0,
                       help='Rank of current process')
    
    args = parser.parse_args()
    
    main(args)