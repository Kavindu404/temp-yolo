import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple


class FeatureExtractor(nn.Module):
    """ResNet backbone feature extractor"""
    
    def __init__(self, backbone: str = 'resnet50'):
        super(FeatureExtractor, self).__init__()
        
        # Load pretrained ResNet model
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            self.feature_dims = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            self.feature_dims = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.feature_dims = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            self.feature_dims = [256, 512, 1024, 2048]
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=True)
            self.feature_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Extract layer blocks for feature pyramid
        self.layer0 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.layer1 = self.backbone.layer1  # 1/4 resolution
        self.layer2 = self.backbone.layer2  # 1/8 resolution
        self.layer3 = self.backbone.layer3  # 1/16 resolution
        self.layer4 = self.backbone.layer4  # 1/32 resolution
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ResNet backbone
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Dictionary of feature maps at different scales
        """
        x0 = self.layer0(x)
        x1 = self.layer1(x0)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32
        
        return {
            'p1': x1,
            'p2': x2, 
            'p3': x3,
            'p4': x4
        }


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    
    def __init__(self, feature_dims: List[int], out_dim: int = 256):
        super(FeaturePyramidNetwork, self).__init__()
        
        # Lateral connections (1x1 convs to reduce channel dimensions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(feature_dims[i], out_dim, kernel_size=1)
            for i in range(len(feature_dims))
        ])
        
        # FPN connections (3x3 convs to smooth features after upsampling and addition)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
            for _ in range(len(feature_dims))
        ])
        
        self.out_dim = out_dim
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the FPN
        
        Args:
            features: Dictionary of feature maps from the backbone
            
        Returns:
            Dictionary of processed feature maps
        """
        # Get feature maps from backbone
        feature_maps = [features[f'p{i+1}'] for i in range(4)]
        
        # Build top-down path
        lat_features = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, feature_maps)]
        
        # Top-down pathway with lateral connections
        fpn_features = [lat_features[-1]]
        for i in range(len(lat_features)-2, -1, -1):
            # Upsample and add
            upsampled = nn.functional.interpolate(
                fpn_features[-1], 
                size=lat_features[i].shape[-2:],
                mode='nearest'
            )
            fpn_features.append(lat_features[i] + upsampled)
        
        # Reverse list to go from fine to coarse
        fpn_features = fpn_features[::-1]
        
        # Apply 3x3 convs for smoothing
        outputs = {}
        for i, (fpn_feature, fpn_conv) in enumerate(zip(fpn_features, self.fpn_convs)):
            outputs[f'p{i+1}'] = fpn_conv(fpn_feature)
            
        return outputs


class YOLOHead(nn.Module):
    """YOLO detection and segmentation head"""
    
    def __init__(self, in_dim: int, num_classes: int, anchors_per_scale: int = 3):
        super(YOLOHead, self).__init__()
        
        # Num outputs: [tx, ty, tw, th, objectness] + num_classes + segmentation mask (1 channel)
        self.num_outputs_per_anchor = 5 + num_classes + 1
        
        # Detection head for each scale
        self.det_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_dim, anchors_per_scale * self.num_outputs_per_anchor, kernel_size=1)
        )
        
        # Segmentation head - takes features from all scales
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_dim * 4, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_dim, in_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_dim // 2, num_classes, kernel_size=1)  # One mask per class
        )
        
        self.anchors_per_scale = anchors_per_scale
        self.num_classes = num_classes
        
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through the YOLO head
        
        Args:
            features: Dictionary of feature maps from the FPN
            
        Returns:
            Tuple of (detection outputs, segmentation outputs)
        """
        # Detection at multiple scales
        detection_outputs = {}
        for scale, feature in features.items():
            output = self.det_head(feature)
            B, _, H, W = output.shape
            
            # Reshape: [B, anchors * outputs, H, W] -> [B, H, W, anchors, outputs]
            output = output.view(B, self.anchors_per_scale, self.num_outputs_per_anchor, H, W)
            output = output.permute(0, 3, 4, 1, 2)  # [B, H, W, anchors, outputs]
            
            detection_outputs[scale] = output
        
        # Segmentation - combine features from all scales
        # Upsample all to a fixed size (e.g., 1/4 of input)
        target_size = features['p1'].shape[-2:]  # Use p1 size as target
        
        seg_features = []
        for scale, feature in features.items():
            if scale != 'p1':
                upsampled = F.interpolate(
                    feature, size=target_size, mode='bilinear', align_corners=False
                )
                seg_features.append(upsampled)
            else:
                seg_features.append(feature)
        
        # Concatenate features from all scales
        seg_input = torch.cat(seg_features, dim=1)
        
        # Generate segmentation masks
        seg_output = self.seg_head(seg_input)
        
        return detection_outputs, seg_output


class YOLOResNet(nn.Module):
    """Complete YOLO model with ResNet backbone, FPN, and detection/segmentation heads"""
    
    def __init__(
        self, 
        backbone: str = 'resnet50', 
        num_classes: int = 80,
        input_size: int = 640,
        fpn_dim: int = 256
    ):
        super(YOLOResNet, self).__init__()
        
        # Backbone
        self.backbone = FeatureExtractor(backbone)
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(self.backbone.feature_dims, fpn_dim)
        
        # YOLO Head
        self.yolo_head = YOLOHead(fpn_dim, num_classes)
        
        # Other parameters
        self.num_classes = num_classes
        self.input_size = input_size
    
    def forward(self, x: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through the YOLOResNet model
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (detection outputs, segmentation outputs)
        """
        # Extract features from backbone
        backbone_features = self.backbone(x)
        
        # Process features through FPN
        fpn_features = self.fpn(backbone_features)
        
        # Generate detections and segmentations
        det_output, seg_output = self.yolo_head(fpn_features)
        
        return det_output, seg_output
