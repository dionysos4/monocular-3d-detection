import numpy as np
import cv2
import torch
import torchvision
from typing import Tuple, List


class Visualizer:
    """
    Provides visualization methods for detection results.
    
    Handles:
    - Visualizing 2D detections with masks and labels
    - Drawing edges and points
    - Projecting and drawing 3D bounding boxes
    """
    
    def __init__(self, detected_color: Tuple[int, int, int] = (228, 26, 28)):
        """
        Initialize visualizer.
        
        Args:
            detected_color: BGR color for detected objects
        """
        self.detected_color = detected_color
        self.point_color = (77, 175, 74)  # Green for special points
    

    def visualize_detections(self, image: np.ndarray, binary_masks, 
                           scores, boxes, labels,
                           coco_categories: dict,
                           plot_label: bool = False) -> np.ndarray:
        """
        Visualize 2D detections with segmentation masks and labels.
        
        Args:
            image: Input image (H, W, 3)
            binary_masks: Binary masks tensor (N, 1, H, W)
            scores: Detection scores
            boxes: Bounding boxes
            labels: Class labels
            coco_categories: COCO category mapping dict
            plot_label: Whether to draw labels and scores
            
        Returns:
            Visualized image with masks and labels
        """
        img_tensor = torch.tensor(image).permute(2, 0, 1)
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        
        # Draw segmentation masks
        uint_img_tensor = (img_tensor * 255).type(torch.uint8)
        local_mask = binary_masks.view(-1, h, w)
        bool_masks = local_mask.type(torch.bool)
        
        colored_array = [self.detected_color for _ in range(bool_masks.shape[0])]
        masked_img = torchvision.utils.draw_segmentation_masks(
            uint_img_tensor, masks=bool_masks, alpha=0.5, colors=colored_array
        )
        masked_img = masked_img.permute(1, 2, 0).numpy()
        
        # Draw labels if requested
        if plot_label:
            for i in range(boxes.shape[0]):
                box = boxes[i].numpy().astype(int)
                label_name = coco_categories[labels[i].item()]
                score = scores[i].item()
                label_text = f"{score:.2f} {label_name}"
                
                # Draw label background
                cv2.rectangle(masked_img, (box[0]-5, box[1]-35), 
                            (box[0]+180, box[1]-5), self.detected_color, -1)
                # Draw label text
                cv2.putText(masked_img, label_text, (box[0], box[1]-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        return masked_img
    

    def visualize_lower_edges(self, binary_masks: np.ndarray, 
                            lower_edges: List[np.ndarray],
                            point_size: int = 1) -> np.ndarray:
        """
        Visualize lower edges on binary masks.
        
        Args:
            binary_masks: Binary masks (N, H, W)
            lower_edges: List of lower edge coordinates
            point_size: Circle radius for edge points
            
        Returns:
            Visualization image
        """
        H, W = binary_masks[0].shape
        edge_img = np.zeros((H, W, 3), dtype=bool)
        
        # Create mask overlay
        for mask in binary_masks:
            edge_img[:, :, 0] = mask | edge_img[:, :, 0]
            edge_img[:, :, 1] = mask | edge_img[:, :, 1]
            edge_img[:, :, 2] = mask | edge_img[:, :, 2]
        
        edge_img = (edge_img.astype(np.uint8) * 255)
        
        # Draw edge points
        for edge in lower_edges:
            for x, y in edge:
                cv2.circle(edge_img, (x, y), point_size, self.detected_color, -1)
        
        return edge_img
    

    def visualize_ransac_edges(self, base_image: np.ndarray,
                              ransac_edges: List[Tuple],
                              point_size: int = 1) -> np.ndarray:
        """
        Visualize RANSAC-fitted edges.
        
        Args:
            base_image: Base image to draw on
            ransac_edges: List of (X, y_pred) tuples from RANSAC
            point_size: Circle radius for points
            
        Returns:
            Image with RANSAC edges drawn
        """
        img_with_ransac = base_image.copy()
        
        for X, y_pred in ransac_edges:
            if X is None or y_pred is None:
                continue
            for x, y in zip(X, y_pred):
                cv2.circle(img_with_ransac, (int(x), int(y)), 
                         point_size, self.point_color, -1)
        
        return img_with_ransac
    

    def visualize_points(self, image: np.ndarray, points: np.ndarray,
                        point_size: int = 1, 
                        color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Draw points on image.
        
        Args:
            image: Input image
            points: Point coordinates (N, 2)
            point_size: Circle radius
            color: Point color (BGR), uses default green if None
            
        Returns:
            Image with points drawn
        """
        if color is None:
            color = self.point_color
        
        point_img = image.copy()
        for point in points:
            x, y = point
            cv2.circle(point_img, (int(x), int(y)), point_size, color, -1)
        
        return point_img
    

    def draw_bbox_wireframe(self, image: np.ndarray, 
                           bbox_points_2d: np.ndarray,
                           color: Tuple[int, int, int] = (0, 255, 0),
                           line_size: int = 1) -> np.ndarray:
        """
        Draw 3D bounding box wireframe on image.
        
        Args:
            image: Input image
            bbox_points_2d: 2D projected bbox corners (8, 2)
            color: Line color (BGR)
            line_size: Line thickness
            
        Returns:
            Image with bbox wireframe drawn
        """
        img_with_lines = image.copy()
        
        # Define connections between 8 corners
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical lines
        ]
        
        for start_idx, end_idx in connections:
            start_point = tuple(bbox_points_2d[start_idx].astype(int))
            end_point = tuple(bbox_points_2d[end_idx].astype(int))
            cv2.line(img_with_lines, start_point, end_point, 
                    color=color, thickness=line_size)
        
        return img_with_lines
    
    
    def visualize_all_bboxes(self, image: np.ndarray,
                            bbox_points_2d: np.ndarray,
                            color: Tuple[int, int, int] = (0, 255, 0),
                            line_size: int = 1) -> np.ndarray:
        """
        Draw all 3D bounding boxes on image.
        
        Args:
            image: Input image
            bbox_points_2d: All projected bbox corners (N, 8, 2)
            color: Line color (BGR)
            line_size: Line thickness
            
        Returns:
            Image with all bboxes drawn
        """
        img_with_bboxes = image.copy()
        
        for bbox_2d in bbox_points_2d:
            img_with_bboxes = self.draw_bbox_wireframe(
                img_with_bboxes, bbox_2d, color, line_size
            )
        
        return img_with_bboxes
