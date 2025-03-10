import os
import torch
import torchvision
import numpy as np
import random
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import json
from typing import Dict, List, Tuple, Union, Optional
import pycocotools.coco as coco
from pycocotools import mask as mask_utils


class COCODataset(Dataset):
    """
    COCO dataset for object detection and segmentation with YOLO model
    Includes mosaic and copy-paste augmentations
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        img_size: int = 640,
        transform: bool = True,
        mosaic_prob: float = 0.5,
        copy_paste_prob: float = 0.5,
        max_objects: int = 100
    ):
        super(COCODataset, self).__init__()
        
        self.data_path = data_path
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.mosaic_prob = mosaic_prob if split == 'train' else 0.0
        self.copy_paste_prob = copy_paste_prob if split == 'train' else 0.0
        self.max_objects = max_objects
        
        # Set up paths
        if split == 'train':
            self.img_dir = os.path.join(data_path, 'train2017')
            self.anno_path = os.path.join(data_path, 'annotations', 'instances_train2017.json')
        elif split == 'val':
            self.img_dir = os.path.join(data_path, 'val2017')
            self.anno_path = os.path.join(data_path, 'annotations', 'instances_val2017.json')
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Load COCO annotations
        self.coco = coco.COCO(self.anno_path)
        self.img_ids = self.coco.getImgIds()
        
        # Get all categories
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.categories.sort(key=lambda x: x['id'])
        
        # Create category to label mapping (continuous indexing from 0)
        self.cat_to_label = {cat['id']: i for i, cat in enumerate(self.categories)}
        self.label_to_cat = {i: cat['id'] for i, cat in enumerate(self.categories)}
        self.num_classes = len(self.categories)
        
        # Pre-load all annotations to speed up mosaic and copy-paste
        self.annotations = self._load_all_annotations()
        
        # Set up transformations
        self.basic_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentations for training
        self.augment = transform and split == 'train'
        if self.augment:
            self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        
        print(f"Loaded {len(self.img_ids)} images from {split} split")
    
    def _load_all_annotations(self) -> Dict[int, Dict]:
        """
        Pre-load all annotations for faster access during training
        
        Returns:
            Dictionary mapping img_id to annotations
        """
        all_annotations = {}
        
        for img_id in self.img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Filter out annotations without segmentation or with no area
            anns = [ann for ann in anns if ann.get('segmentation') and ann.get('area', 0) > 0]
            
            all_annotations[img_id] = {
                'img_info': img_info,
                'anns': anns
            }
        
        return all_annotations
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, target, mask)
                - image: Tensor of shape (3, H, W)
                - target: Tensor of shape (num_objects, 5) - class, x, y, w, h in normalized coords
                - mask: Tensor of shape (H, W) with class indices
        """
        img_id = self.img_ids[idx]
        
        # Decide whether to use mosaic augmentation
        if self.transform and random.random() < self.mosaic_prob:
            image, target, mask = self._load_mosaic(idx)
        else:
            image, target, mask = self._load_image_and_annotations(img_id)
        
        # Apply copy-paste augmentation
        if self.transform and random.random() < self.copy_paste_prob:
            image, target, mask = self._apply_copy_paste(image, target, mask)
        
        # Apply other augmentations
        if self.augment:
            # Apply color jitter
            if random.random() < 0.5:
                image = self.color_jitter(image)
            
            # Random horizontal flip
            if random.random() < 0.5:
                image = T.functional.hflip(image)
                mask = T.functional.hflip(mask)
                
                # Flip boxes
                if target.size(0) > 0:
                    target[:, 1] = 1.0 - target[:, 1]  # Flip x center
        
        # Convert image to tensor and normalize
        if not isinstance(image, torch.Tensor):
            image = self.basic_transform(image)
        
        # Ensure mask is a long tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.long)
        
        return image, target, mask
    
    def _load_image_and_annotations(
        self, 
        img_id: int, 
        img_size: Optional[int] = None
    ) -> Tuple[Union[Image.Image, torch.Tensor], torch.Tensor, np.ndarray]:
        """
        Load an image and its annotations
        
        Args:
            img_id: Image ID
            img_size: Image size to resize to (optional)
            
        Returns:
            Tuple of (image, target, mask)
        """
        # Get image info and annotations
        img_info = self.annotations[img_id]['img_info']
        anns = self.annotations[img_id]['anns']
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        
        # Get original size
        orig_w, orig_h = img.size
        
        # Resize image
        img_size = img_size or self.img_size
        img = img.resize((img_size, img_size))
        
        # Prepare target: [class_idx, x_center, y_center, width, height]
        target = []
        mask = np.zeros((img_size, img_size), dtype=np.int64)
        
        for ann in anns:
            # Get category label (continuous indexing from 0)
            cat_id = ann['category_id']
            label = self.cat_to_label[cat_id]
            
            # Get bounding box (COCO format: x, y, w, h)
            x, y, w, h = ann['bbox']
            
            # Convert to center format and normalize
            x_center = (x + w / 2) / orig_w
            y_center = (y + h / 2) / orig_h
            width = w / orig_w
            height = h / orig_h
            
            # Add to target
            target.append([label, x_center, y_center, width, height])
            
            # Get segmentation mask
            if 'segmentation' in ann:
                # Convert segmentation to binary mask
                if isinstance(ann['segmentation'], list):  # Polygon format
                    rle = mask_utils.frPyObjects(ann['segmentation'], orig_h, orig_w)
                    seg_mask = mask_utils.decode(rle)
                    if len(seg_mask.shape) > 2:
                        seg_mask = seg_mask.sum(axis=2) > 0
                else:  # RLE format
                    seg_mask = mask_utils.decode(ann['segmentation'])
                
                # Resize mask to img_size
                seg_mask = Image.fromarray(seg_mask.astype(np.uint8) * 255)
                seg_mask = seg_mask.resize((img_size, img_size))
                seg_mask = np.array(seg_mask) > 127
                
                # Add to instance segmentation mask with category label
                mask[seg_mask] = label + 1  # Add 1 to differentiate from background (0)
        
        # Convert target to tensor
        target = torch.tensor(target, dtype=torch.float32)
        
        return img, target, mask
    
    def _load_mosaic(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Load mosaic augmentation
        
        Args:
            idx: Index of the center image
            
        Returns:
            Tuple of (mosaic_img, mosaic_target, mosaic_mask)
        """
        # Get 4 random images
        indices = [idx] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        
        # Create mosaic of size (2*img_size, 2*img_size) and then resize to img_size
        mosaic_img_size = self.img_size * 2
        mosaic_img = Image.new('RGB', (mosaic_img_size, mosaic_img_size))
        mosaic_mask = np.zeros((mosaic_img_size, mosaic_img_size), dtype=np.int64)
        
        # Coordinates for placing the 4 images
        grid = [
            (0, 0),
            (mosaic_img_size // 2, 0),
            (0, mosaic_img_size // 2),
            (mosaic_img_size // 2, mosaic_img_size // 2)
        ]
        
        # Initialize mosaic targets
        mosaic_targets = []
        
        # Place the 4 images
        for i, img_idx in enumerate(indices):
            img_id = self.img_ids[img_idx]
            img, target, mask = self._load_image_and_annotations(img_id, img_size=mosaic_img_size // 2)
            
            # Place image
            x_offset, y_offset = grid[i]
            if isinstance(img, torch.Tensor):
                # Convert back to PIL for pasting
                img = T.ToPILImage()(img)
            
            mosaic_img.paste(img, (x_offset, y_offset))
            
            # Place mask
            mosaic_mask[y_offset:y_offset + mosaic_img_size // 2, 
                        x_offset:x_offset + mosaic_img_size // 2] = mask
            
            # Adjust target coordinates
            if target.size(0) > 0:
                # Adjust coordinates to mosaic space
                target[:, 1] = (target[:, 1] * (mosaic_img_size // 2) + x_offset) / mosaic_img_size
                target[:, 2] = (target[:, 2] * (mosaic_img_size // 2) + y_offset) / mosaic_img_size
                target[:, 3] = target[:, 3] * (mosaic_img_size // 2) / mosaic_img_size
                target[:, 4] = target[:, 4] * (mosaic_img_size // 2) / mosaic_img_size
                
                mosaic_targets.append(target)
        
        # Combine all targets
        if len(mosaic_targets) > 0:
            mosaic_targets = torch.cat(mosaic_targets, dim=0)
        else:
            mosaic_targets = torch.zeros((0, 5), dtype=torch.float32)
        
        # Resize mosaic to self.img_size
        mosaic_img = mosaic_img.resize((self.img_size, self.img_size))
        
        # Resize mosaic mask
        mosaic_mask_pil = Image.fromarray(mosaic_mask.astype(np.uint8))
        mosaic_mask_pil = mosaic_mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)
        mosaic_mask = np.array(mosaic_mask_pil)
        
        # Adjust target coordinates for resized mosaic
        if mosaic_targets.size(0) > 0:
            # No need to adjust coordinates as they're already normalized
            pass
        
        return mosaic_img, mosaic_targets, mosaic_mask
    
    def _apply_copy_paste(
        self, 
        image: Union[Image.Image, torch.Tensor], 
        target: torch.Tensor, 
        mask: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Union[Image.Image, torch.Tensor], torch.Tensor, Union[np.ndarray, torch.Tensor]]:
        """
        Apply copy-paste augmentation
        
        Args:
            image: Input image
            target: Target tensor (class, x, y, w, h)
            mask: Segmentation mask
            
        Returns:
            Tuple of (augmented_img, augmented_target, augmented_mask)
        """
        # Convert to PIL if tensor
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            orig_image = T.ToPILImage()(image)
        else:
            orig_image = image
        
        # Convert mask to numpy if tensor
        is_mask_tensor = isinstance(mask, torch.Tensor)
        if is_mask_tensor:
            orig_mask = mask.numpy()
        else:
            orig_mask = mask
        
        # Create new image and mask
        new_image = orig_image.copy()
        new_mask = orig_mask.copy()
        
        # Get a random image to copy objects from
        paste_idx = random.randint(0, len(self.img_ids) - 1)
        paste_img_id = self.img_ids[paste_idx]
        paste_img, paste_target, paste_mask = self._load_image_and_annotations(paste_img_id)
        
        # Convert paste_img to PIL if tensor
        if isinstance(paste_img, torch.Tensor):
            paste_img = T.ToPILImage()(paste_img)
        
        # Only copy if there are objects in the paste image
        if paste_target.size(0) > 0:
            # Randomly select objects to copy (up to 3)
            num_to_copy = min(3, paste_target.size(0))
            copy_indices = random.sample(range(paste_target.size(0)), num_to_copy)
            
            # Create new targets
            new_targets = [target]
            
            for idx in copy_indices:
                # Get object class and bounding box
                cls, x_center, y_center, width, height = paste_target[idx]
                
                # Convert to pixel coordinates
                x1 = int((x_center - width / 2) * self.img_size)
                y1 = int((y_center - height / 2) * self.img_size)
                x2 = int((x_center + width / 2) * self.img_size)
                y2 = int((y_center + height / 2) * self.img_size)
                
                # Get object mask
                obj_mask = paste_mask == (cls.item() + 1)
                if obj_mask.sum() == 0:
                    continue  # Skip if no valid mask
                
                # Create a cropped mask for the object
                cropped_mask = np.zeros_like(obj_mask)
                cropped_mask[y1:y2, x1:x2] = obj_mask[y1:y2, x1:x2]
                
                # Get random placement coordinates
                new_x_center = random.uniform(width/2, 1 - width/2)
                new_y_center = random.uniform(height/2, 1 - height/2)
                
                # Convert to pixel coordinates
                new_x1 = int((new_x_center - width / 2) * self.img_size)
                new_y1 = int((new_y_center - height / 2) * self.img_size)
                new_x2 = int((new_x_center + width / 2) * self.img_size)
                new_y2 = int((new_y_center + height / 2) * self.img_size)
                
                # Create new target
                new_target = torch.tensor([[cls, new_x_center, new_y_center, width, height]], 
                                         dtype=torch.float32)
                new_targets.append(new_target)
                
                # Crop the object from the paste image
                obj_img = paste_img.crop((x1, y1, x2, y2))
                
                # Create alpha mask for blending
                alpha_mask = Image.fromarray((cropped_mask[y1:y2, x1:x2] * 255).astype(np.uint8))
                
                # Paste the object to the new location
                new_image.paste(obj_img, (new_x1, new_y1), alpha_mask)
                
                # Update segmentation mask
                new_mask_region = new_mask[new_y1:new_y2, new_x1:new_x2]
                cropped_obj_mask = cropped_mask[y1:y2, x1:x2]
                
                # Only paste where the object mask is True
                if new_mask_region.shape == cropped_obj_mask.shape:
                    new_mask_region[cropped_obj_mask] = cls + 1
            
            # Combine all targets
            target = torch.cat(new_targets, dim=0)
        
        # Convert back to tensor if needed
        if is_tensor:
            image = self.basic_transform(new_image)
        else:
            image = new_image
        
        if is_mask_tensor:
            mask = torch.tensor(new_mask, dtype=torch.long)
        else:
            mask = new_mask
        
        return image, target, mask
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Custom collate function for batching
        
        Args:
            batch: List of (image, target, mask) tuples
            
        Returns:
            Tuple of (batched_images, list_of_targets, batched_masks)
        """
        images, targets, masks = zip(*batch)
        
        # Stack images and masks
        batched_images = torch.stack(images)
        batched_masks = torch.stack(masks)
        
        # Return images, list of targets, and masks
        return batched_images, targets, batched_masks
