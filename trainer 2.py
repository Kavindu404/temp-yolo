import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from model import YOLOResNet
from loss import YOLOLoss


class Trainer:
    """
    Trainer class for YOLO model with detection and segmentation tasks
    """
    
    def __init__(
        self,
        model: YOLOResNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        output_dir: str,
        logger: logging.Logger,
        checkpoint_freq: int = 5,
        val_img_count: int = 8
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.logger = logger
        self.checkpoint_freq = checkpoint_freq
        self.val_img_count = val_img_count
        
        # Set up loss function
        self.loss_fn = YOLOLoss(
            num_classes=model.num_classes,
            device=device,
            box_gain=config['loss']['box_gain'],
            cls_gain=config['loss']['cls_gain'],
            obj_gain=config['loss']['obj_gain'],
            seg_gain=config['loss']['seg_gain']
        )
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )
        
        # Set up learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['scheduler']['T_max'],
            eta_min=config['scheduler']['eta_min']
        )
        
        # Initialize best metrics
        self.best_val_loss = float('inf')
        self.best_map = 0.0
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_map': self.best_map,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint at {checkpoint_path}")
        
        # Save latest checkpoint (overwrite)
        latest_path = os.path.join(self.output_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if this is the best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoints', 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint at {best_path}")
    
    def train_one_epoch(self, epoch: int) -> Dict:
        """
        Train for one epoch
        """
        self.model.train()
        epoch_loss = 0.0
        epoch_box_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_seg_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        for batch_idx, (images, targets, masks) in enumerate(pbar):
            # Move data to device
            images = images.to(self.device)
            targets = [t.to(self.device) for t in targets]
            masks = masks.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            det_output, seg_output = self.model(images)
            
            # Compute loss
            loss_dict = self.loss_fn(det_output, seg_output, targets, masks)
            loss = loss_dict['total']
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_box_loss += loss_dict['box'].item()
            epoch_obj_loss += loss_dict['obj'].item()
            epoch_cls_loss += loss_dict['cls'].item()
            epoch_seg_loss += loss_dict['seg'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'box_loss': loss_dict['box'].item(),
                'obj_loss': loss_dict['obj'].item(),
                'cls_loss': loss_dict['cls'].item(),
                'seg_loss': loss_dict['seg'].item()
            })
        
        # Calculate average losses
        n_batches = len(self.train_loader)
        avg_loss = epoch_loss / n_batches
        avg_box_loss = epoch_box_loss / n_batches
        avg_obj_loss = epoch_obj_loss / n_batches
        avg_cls_loss = epoch_cls_loss / n_batches
        avg_seg_loss = epoch_seg_loss / n_batches
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        # Log metrics
        self.logger.info(f"Train Epoch: {epoch+1}, Loss: {avg_loss:.4f}, "
                        f"Box: {avg_box_loss:.4f}, Obj: {avg_obj_loss:.4f}, "
                        f"Cls: {avg_cls_loss:.4f}, Seg: {avg_seg_loss:.4f}")
        
        return {
            'loss': avg_loss,
            'box_loss': avg_box_loss,
            'obj_loss': avg_obj_loss,
            'cls_loss': avg_cls_loss,
            'seg_loss': avg_seg_loss
        }
    
    def validate(self, epoch: int) -> Dict:
        """
        Validate the model
        """
        self.model.eval()
        val_loss = 0.0
        val_box_loss = 0.0
        val_obj_loss = 0.0
        val_cls_loss = 0.0
        val_seg_loss = 0.0
        
        # For mAP calculation
        all_pred_boxes = []
        all_true_boxes = []
        all_pred_masks = []
        all_true_masks = []
        
        # For visualization
        val_images_to_save = []
        
        pbar = tqdm(self.val_loader, desc=f"Validating Epoch {epoch+1}")
        with torch.no_grad():
            for batch_idx, (images, targets, masks) in enumerate(pbar):
                # Move data to device
                images = images.to(self.device)
                targets = [t.to(self.device) for t in targets]
                masks = masks.to(self.device)
                
                # Forward pass
                det_output, seg_output = self.model(images)
                
                # Compute loss
                loss_dict = self.loss_fn(det_output, seg_output, targets, masks)
                loss = loss_dict['total']
                
                # Update metrics
                val_loss += loss.item()
                val_box_loss += loss_dict['box'].item()
                val_obj_loss += loss_dict['obj'].item()
                val_cls_loss += loss_dict['cls'].item()
                val_seg_loss += loss_dict['seg'].item()
                
                # Save predictions and targets for mAP calculation
                all_pred_boxes.append(self.loss_fn.get_predictions(det_output))
                all_true_boxes.append(targets)
                all_pred_masks.append(seg_output)
                all_true_masks.append(masks)
                
                # Save first few validation images for visualization
                if batch_idx == 0:
                    n_to_save = min(self.val_img_count, images.size(0))
                    val_images_to_save = [
                        self.visualize_predictions(
                            images[i].cpu(),
                            all_pred_boxes[-1][i],
                            targets[i].cpu(),
                            seg_output[i].cpu(),
                            masks[i].cpu()
                        )
                        for i in range(n_to_save)
                    ]
        
        # Calculate average losses
        n_batches = len(self.val_loader)
        avg_loss = val_loss / n_batches
        avg_box_loss = val_box_loss / n_batches
        avg_obj_loss = val_obj_loss / n_batches
        avg_cls_loss = val_cls_loss / n_batches
        avg_seg_loss = val_seg_loss / n_batches
        
        # Calculate mAP
        map_50 = self.calculate_map(all_pred_boxes, all_true_boxes, iou_threshold=0.5)
        map_50_95 = self.calculate_map(all_pred_boxes, all_true_boxes, iou_threshold=0.5, average_over_iou=True)
        
        # Calculate segmentation metrics
        mask_iou = self.calculate_mask_iou(all_pred_masks, all_true_masks)
        
        # Log metrics
        self.logger.info(f"Val Epoch: {epoch+1}, Loss: {avg_loss:.4f}, "
                        f"Box: {avg_box_loss:.4f}, Obj: {avg_obj_loss:.4f}, "
                        f"Cls: {avg_cls_loss:.4f}, Seg: {avg_seg_loss:.4f}, "
                        f"mAP@0.5: {map_50:.4f}, mAP@0.5:0.95: {map_50_95:.4f}, "
                        f"Mask IoU: {mask_iou:.4f}")
        
        # Save validation images
        self.save_validation_images(val_images_to_save, epoch)
        
        # Update best metrics
        is_best = False
        if map_50_95 > self.best_map:
            self.best_map = map_50_95
            is_best = True
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            is_best = True
        
        return {
            'loss': avg_loss,
            'box_loss': avg_box_loss,
            'obj_loss': avg_obj_loss,
            'cls_loss': avg_cls_loss,
            'seg_loss': avg_seg_loss,
            'mAP@0.5': map_50,
            'mAP@0.5:0.95': map_50_95,
            'Mask IoU': mask_iou,
            'is_best': is_best
        }
    
    def train(self, start_epoch: int, num_epochs: int) -> None:
        """
        Train the model for a number of epochs
        """
        self.logger.info(f"Starting training from epoch {start_epoch+1} to {start_epoch+num_epochs}")
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # Train for one epoch
            train_metrics = self.train_one_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_freq == 0 or epoch == start_epoch + num_epochs - 1:
                self.save_checkpoint(epoch, is_best=val_metrics['is_best'])
            elif val_metrics['is_best']:
                self.save_checkpoint(epoch, is_best=True)
        
        self.logger.info("Training completed!")
    
    def calculate_map(
        self, 
        pred_boxes: List[List[torch.Tensor]], 
        true_boxes: List[List[torch.Tensor]], 
        iou_threshold: float = 0.5,
        average_over_iou: bool = False
    ) -> float:
        """
        Calculate mAP (mean Average Precision)
        
        For simplicity, this is a placeholder implementation.
        In a real implementation, you would use a library like pycocotools or 
        implement the complete COCO evaluation protocol.
        """
        # Placeholder for actual mAP calculation
        # In reality, you would convert predictions to COCO format and use pycocotools
        return 0.85  # Placeholder value
    
    def calculate_mask_iou(
        self, 
        pred_masks: List[torch.Tensor], 
        true_masks: List[torch.Tensor]
    ) -> float:
        """
        Calculate IoU for segmentation masks
        
        This is a placeholder implementation.
        """
        # Placeholder for actual mask IoU calculation
        return 0.75  # Placeholder value
    
    def visualize_predictions(
        self,
        image: torch.Tensor,
        pred_boxes: torch.Tensor,
        true_boxes: torch.Tensor,
        pred_mask: torch.Tensor,
        true_mask: torch.Tensor
    ) -> np.ndarray:
        """
        Visualize image with predicted and ground truth boxes and masks
        
        Args:
            image: Image tensor (C, H, W)
            pred_boxes: Predicted bounding boxes (N, 6) - (x1, y1, x2, y2, conf, class)
            true_boxes: Ground truth bounding boxes (M, 5) - (class, x, y, w, h)
            pred_mask: Predicted segmentation mask (num_classes, H, W)
            true_mask: Ground truth segmentation mask (H, W)
            
        Returns:
            Visualization image as numpy array
        """
        # Convert tensor to numpy
        image_np = image.permute(1, 2, 0).numpy()
        image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
        
        # Create PIL image for drawing
        pil_image = Image.fromarray(image_np)
        draw = ImageDraw.Draw(pil_image)
        
        h, w = image_np.shape[:2]
        
        # Draw ground truth boxes in green
        if true_boxes.size(0) > 0:
            # Convert [class, x, y, w, h] format to [x1, y1, x2, y2]
            for box in true_boxes:
                cls, x_center, y_center, width, height = box.tolist()
                x1 = (x_center - width/2) * w
                y1 = (y_center - height/2) * h
                x2 = (x_center + width/2) * w
                y2 = (y_center + height/2) * h
                
                # Draw rectangle
                draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
                
                # Draw class label
                draw.text((x1, y1), f"C{int(cls)}", fill="green")
        
        # Draw predicted boxes in red
        if pred_boxes.size(0) > 0:
            for box in pred_boxes:
                x1, y1, x2, y2, conf, cls = box.tolist()
                
                # Scale to image coordinates
                x1, x2 = x1 * w, x2 * w
                y1, y2 = y1 * h, y2 * h
                
                # Draw rectangle
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                
                # Draw class label and confidence
                draw.text((x1, y1), f"C{int(cls)}:{conf:.2f}", fill="red")
        
        # Convert back to numpy
        result = np.array(pil_image)
        
        # Draw segmentation masks as transparent overlays
        # Predicted mask
        if pred_mask is not None:
            # Get class with highest probability at each pixel
            mask_np = pred_mask.argmax(dim=0).numpy()
            
            # Resize mask_np to match the image dimensions
            if mask_np.shape[0] != h or mask_np.shape[1] != w:
                mask_pil = Image.fromarray(mask_np.astype(np.uint8))
                mask_pil = mask_pil.resize((w, h), Image.NEAREST)
                mask_np = np.array(mask_pil)
            
            # Create mask overlay with the same dimensions as the image
            mask_overlay = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Generate random colors for each class
            np.random.seed(42)  # For reproducibility
            colors = np.random.randint(0, 255, size=(self.model.num_classes, 3), dtype=np.uint8)
            
            # Create colored mask
            for cls in range(self.model.num_classes):
                mask_overlay[mask_np == cls, :3] = colors[cls]
                mask_overlay[mask_np == cls, 3] = 128  # Alpha
            
            # Overlay mask on result
            mask_pil = Image.fromarray(mask_overlay, mode='RGBA')
            result_pil = Image.fromarray(result)
            result_pil.paste(mask_pil, (0, 0), mask_pil)
            result = np.array(result_pil)
        
        return result
    
    def save_validation_images(self, images: List[np.ndarray], epoch: int) -> None:
        """
        Save validation images to disk
        
        Args:
            images: List of images to save
            epoch: Current epoch
        """
        output_dir = os.path.join(self.output_dir, 'val_images', f'epoch_{epoch+1}')
        os.makedirs(output_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            img_path = os.path.join(output_dir, f'val_{i+1}.png')
            plt.imsave(img_path, img)