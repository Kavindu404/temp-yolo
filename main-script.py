#!/usr/bin/env python3
import os
import argparse
import torch
from datetime import datetime

from config import get_config
from model import YOLOResNet
from trainer import Trainer
from dataset import COCODataset
from utils import setup_logger, seed_everything

def parse_args():
    parser = argparse.ArgumentParser(description='Train a YOLO model with ResNet backbone')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                        help='ResNet backbone to use')
    parser.add_argument('--data_path', type=str, required=True, help='Path to COCO dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint_freq', type=int, default=5, help='Checkpoint save frequency (epochs)')
    parser.add_argument('--val_img_count', type=int, default=8, help='Number of validation images to save')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='runs/exp', help='Output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'val_images'), exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.output_dir)
    logger.info(f"Arguments: {args}")
    
    # Set random seed
    seed_everything(args.seed)
    
    # Load config
    config = get_config(args.config)
    config.update(vars(args))
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = COCODataset(
        data_path=args.data_path,
        split='train',
        transform=True,
        mosaic_prob=config['augmentation']['mosaic_prob'],
        copy_paste_prob=config['augmentation']['copy_paste_prob']
    )
    
    val_dataset = COCODataset(
        data_path=args.data_path,
        split='val',
        transform=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    
    # Create model
    model = YOLOResNet(
        backbone=args.backbone,
        num_classes=train_dataset.num_classes,
        input_size=config['model']['input_size']
    )
    model = model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=args.output_dir,
        logger=logger,
        checkpoint_freq=args.checkpoint_freq,
        val_img_count=args.val_img_count
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Train model
    trainer.train(start_epoch, args.epochs)

if __name__ == '__main__':
    main()
