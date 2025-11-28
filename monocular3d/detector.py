import torch
import torchvision
from typing import Tuple


class ObjectDetector:
    """
    Manages 2D object detection using pre-trained models.
    
    Attributes:
        model_name: Name of the detection model (currently only "mask_rcnn")
        model: Loaded PyTorch detection model
    """
    
    # COCO dataset category mapping
    COCO_CATEGORIES = {
        0: '__background__', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
        5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 12: 'N/A', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
        22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'N/A',
        27: 'backpack', 28: 'umbrella', 29: 'N/A', 30: 'N/A', 31: 'handbag',
        32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
        37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
        41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
        45: 'N/A', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
        50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
        55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
        60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant',
        65: 'bed', 66: 'N/A', 67: 'dining table', 68: 'N/A', 69: 'N/A'
    }
    

    def __init__(self, model_name: str = "mask_rcnn"):
        """
        Initialize the object detector.
        
        Args:
            model_name: Name of the detection model to use
        """
        self.model_name = model_name
        self.model = self._load_model()
    

    def _load_model(self):
        """Load the specified detection model."""
        if self.model_name == "mask_rcnn":
            return self._load_mask_rcnn()
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
    

    def _load_mask_rcnn(self):
        """Load pre-trained Mask R-CNN model."""
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        return model
    

    def predict(self, image, detection_threshold: float = 0.7) -> Tuple:
        """
        Run 2D object detection on the input image.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            detection_threshold: Confidence threshold for detections
            
        Returns:
            Tuple of (scores, labels, masks, boxes)
        """
        img_tensor = torch.tensor(image).permute(2, 0, 1)
        if torch.cuda.is_available():
            img_tensor = img_tensor.to("cuda")
        
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]
        
        # Filter by confidence threshold
        high_conf_indices = predictions["scores"] > detection_threshold
        predictions = {k: v[high_conf_indices] for k, v in predictions.items()}
        
        # Move to CPU if needed
        if torch.cuda.is_available():
            predictions = {k: v.cpu() for k, v in predictions.items()}
        
        return (predictions["scores"], predictions["labels"], 
                predictions["masks"], predictions["boxes"])
    
    
    def filter_by_categories(self, scores, labels, masks, boxes, categories):
        """
        Filter detections to keep only specified categories.
        
        Args:
            scores: Detection confidence scores
            labels: Detection class labels
            masks: Detection masks
            boxes: Detection bounding boxes
            categories: List of category names to keep
            
        Returns:
            Filtered tuple of (scores, labels, masks, boxes)
        """
        target_indices = [
            i for i, label in enumerate(labels) 
            if self.COCO_CATEGORIES[label.item()] in categories
        ]
        
        return (scores[target_indices], labels[target_indices],
                masks[target_indices], boxes[target_indices])
