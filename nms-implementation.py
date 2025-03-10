import torch
from torchvision.ops import nms as torch_nms


def batched_nms(boxes, scores, class_ids, iou_threshold=0.5, score_threshold=0.05, max_detections=300):
    """
    Apply non-maximum suppression to avoid detecting too many overlapping bounding boxes.
    
    Args:
        boxes: Tensor of shape [N, 4] with bounding boxes in format [x1, y1, x2, y2]
        scores: Tensor of shape [N] with confidence scores
        class_ids: Tensor of shape [N] with class IDs
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score threshold
        max_detections: Maximum number of detections to keep
        
    Returns:
        Tensor of indices of kept boxes
    """
    # Filter out boxes with low confidence
    mask = scores > score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if boxes.shape[0] == 0:
        return torch.zeros(0, device=boxes.device, dtype=torch.int64)
    
    # Apply per-class NMS
    # For each class, apply NMS and keep track of selected indices
    keep_indices = []
    unique_classes = torch.unique(class_ids)
    
    for cls in unique_classes:
        cls_mask = class_ids == cls
        # Apply NMS for this class
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        
        # Use torchvision's NMS implementation
        cls_keep = torch_nms(cls_boxes, cls_scores, iou_threshold)
        
        # Map back to original indices
        cls_indices = torch.nonzero(cls_mask, as_tuple=True)[0]
        keep_indices.append(cls_indices[cls_keep])
    
    # Combine all kept indices
    if keep_indices:
        keep = torch.cat(keep_indices)
        
        # Sort by score and limit to max_detections
        _, order = scores[keep].sort(descending=True)
        keep = keep[order]
        
        # Limit to max_detections
        if len(keep) > max_detections:
            keep = keep[:max_detections]
            
        return keep[mask.nonzero(as_tuple=True)[0][keep]]
    else:
        return torch.zeros(0, device=boxes.device, dtype=torch.int64)


def multiclass_nms(pred_boxes, pred_scores, pred_classes, iou_threshold=0.5, score_threshold=0.05, max_detections=300):
    """
    Perform NMS on predictions with multiple classes
    
    Args:
        pred_boxes: Tensor of shape [N, 4] with bounding boxes in format [x1, y1, x2, y2]
        pred_scores: Tensor of shape [N] with confidence scores
        pred_classes: Tensor of shape [N] with class predictions
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score threshold
        max_detections: Maximum number of detections to keep
        
    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_classes)
    """
    keep = batched_nms(
        pred_boxes, 
        pred_scores, 
        pred_classes, 
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections
    )
    
    if len(keep) == 0:
        return torch.zeros((0, 4), device=pred_boxes.device), \
               torch.zeros(0, device=pred_scores.device), \
               torch.zeros(0, device=pred_classes.device, dtype=pred_classes.dtype)
    
    return pred_boxes[keep], pred_scores[keep], pred_classes[keep]


def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.05, method='gaussian'):
    """
    Implementation of Soft-NMS which decays scores of overlapping boxes instead of removing them
    
    Args:
        boxes: Tensor of shape [N, 4] with bounding boxes in format [x1, y1, x2, y2]
        scores: Tensor of shape [N] with confidence scores
        iou_threshold: IoU threshold for NMS (used only with 'linear' method)
        sigma: Parameter for gaussian decay
        score_threshold: Minimum score threshold
        method: 'linear' or 'gaussian' decay function
        
    Returns:
        Tuple of (kept_boxes, kept_scores)
    """
    boxes = boxes.detach().clone()
    scores = scores.detach().clone()
    
    # Sort boxes by scores
    _, order = scores.sort(descending=True)
    boxes = boxes[order]
    scores = scores[order]
    
    keep_boxes = []
    keep_scores = []
    
    while boxes.shape[0] > 0:
        # Add max score box to output
        max_box = boxes[0:1]
        max_score = scores[0:1]
        keep_boxes.append(max_box)
        keep_scores.append(max_score)
        
        # Stop if only one box is left
        if boxes.shape[0] == 1:
            break
        
        # Compute IoU of the max box with the rest
        ious = box_iou(max_box, boxes[1:])
        
        # Update scores based on IoU
        if method == 'linear':
            # Linear soft-NMS
            decay = torch.ones_like(ious)
            decay[ious > iou_threshold] = 1 - ious[ious > iou_threshold]
        else:
            # Gaussian soft-NMS
            decay = torch.exp(-(ious * ious) / sigma)
        
        scores[1:] = scores[1:] * decay.squeeze()
        
        # Remove low scoring boxes
        remain = (scores[1:] > score_threshold).nonzero().squeeze(1) + 1
        if remain.size(0) == 0:
            break
            
        # Update boxes and scores
        boxes = boxes[remain]
        scores = scores[remain]
    
    # Concatenate results
    if keep_boxes and keep_scores:
        keep_boxes = torch.cat(keep_boxes, dim=0)
        keep_scores = torch.cat(keep_scores, dim=0)
        return keep_boxes, keep_scores
    else:
        return torch.zeros((0, 4), device=boxes.device), torch.zeros(0, device=scores.device)


def box_iou(box1, box2):
    """
    Compute IoU between boxes
    
    Args:
        box1: Tensor of shape [N, 4] with boxes in format [x1, y1, x2, y2]
        box2: Tensor of shape [M, 4] with boxes in format [x1, y1, x2, y2]
        
    Returns:
        Tensor of shape [N, M] with IoU values
    """
    # Get box areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Get intersection coordinates
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # Left-top [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # Right-bottom [N,M,2]
    
    # Calculate intersection area
    wh = (rb - lt).clamp(min=0)  # Width-height [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # Intersection area [N,M]
    
    # Calculate IoU
    union = area1[:, None] + area2 - inter
    iou = inter / union
    
    return iou


# Usage example in loss.py
def apply_nms_to_predictions(predictions, iou_threshold=0.5, score_threshold=0.25, max_detections=100):
    """
    Apply NMS to predictions
    
    Args:
        predictions: Tensor of shape [N, 6] with boxes in format [x1, y1, x2, y2, confidence, class]
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score threshold
        max_detections: Maximum number of detections to keep
        
    Returns:
        Filtered predictions after NMS
    """
    if predictions.size(0) == 0:
        return predictions
    
    # Extract components
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    classes = predictions[:, 5].int()
    
    # Apply multiclass NMS
    filtered_boxes, filtered_scores, filtered_classes = multiclass_nms(
        boxes, 
        scores, 
        classes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections
    )
    
    # Combine results
    if filtered_boxes.size(0) > 0:
        filtered_predictions = torch.cat([
            filtered_boxes,
            filtered_scores.unsqueeze(-1),
            filtered_classes.float().unsqueeze(-1)
        ], dim=1)
        return filtered_predictions
    else:
        return torch.zeros((0, 6), device=predictions.device)
