import numpy as np
from typing import List, Dict

from .detector import ObjectDetector
from .geometry import GeometryProcessor
from .bbox import BBoxProcessor
from .visualization import Visualizer


class MonoDetection:
    """
    Monocular 3D Object Detection - Main Facade
    
    Estimates 3D objects from a single calibrated camera and a known
    transformation between camera and plane coordinate system.
    
    Modular architecture with separate components for:
    - 2D object detection (ObjectDetector)
    - Geometric processing (GeometryProcessor)
    - Bounding box computation (BBoxProcessor)
    - Visualization (Visualizer)
    
    Attributes:
        model_name: Name of the 2D detection model (default: "mask_rcnn")
    
    Example:
        >>> detector = MonoDetection()
        >>> bbox_params = detector.predict(image, K, T_cam_plane, 
        ...                                categories=["boat"], fixed_depth=3.0)
        >>> result_img = detector.get_2d_detected_img()
    """
    
    def __init__(self, model_name: str = "mask_rcnn"):
        """
        Initialize the monocular 3D object detector.
        
        Args:
            model_name: Name of the 2D detection model to use (default: "mask_rcnn")
        """
        self.model_name = model_name
        
        # Initialize modular components
        self.detector = ObjectDetector(model_name)
        self.geometry = GeometryProcessor()
        self.bbox_processor = BBoxProcessor()
        self.visualizer = Visualizer()
        
        # State variables (populated during prediction)
        self.image = None
        self.K = None
        self.T_cam_plane = None
        self.mask_rcnn_image = None
        self.binary_masks = None
        self.lower_edges = None
        self.upper_points = None
        self.ransac_lower_edges = None
        self.bbox_parameters = None
        
        # Expose COCO categories for compatibility
        self.COCO_INSTANCE_CATEGORY_DICT = ObjectDetector.COCO_CATEGORIES
        self.detected_color = self.visualizer.detected_color


    def predict(self, image: np.ndarray, K: np.ndarray, T_cam_plane: np.ndarray,
                categories: List[str] = None, fixed_depth: float = 3.0,
                detection_threshold: float = 0.7) -> List[Dict]:
        """
        Predict 3D bounding boxes from the input image.
        
        This is the main API method that orchestrates the entire detection
        and reconstruction pipeline:
        1. Run 2D object detection
        2. Filter by target categories
        3. Extract geometric features from masks
        4. Perform RANSAC edge fitting
        5. Reconstruct 3D points
        6. Compute object heights
        7. Build 3D bounding boxes
        
        Args:
            image: Input image from monocular camera (H, W, 3)
            K: Camera intrinsic matrix (3, 3)
            T_cam_plane: Transformation matrix from plane to camera coords (4, 4)
            categories: List of object categories to detect (default: ["boat"])
                Possible categories: 'person', 'bicycle', 'car', 'motorcycle', 
                'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table'
            fixed_depth: Fixed depth value for all objects (meters)
            detection_threshold: Confidence threshold for 2D detections (0-1)
        
        Returns:
            List of bounding box parameter dicts, each containing:
                - x, y, z: Center coordinates in plane coordinate system
                - length, height, width: Box dimensions
                - yaw: Rotation around y-axis (radians)
        """
        if categories is None:
            categories = ["boat"]
        
        # Store inputs for later use in visualization
        self.image = image
        self.K = K
        self.T_cam_plane = T_cam_plane
        
        # Step 1: Run 2D object detection
        scores, labels, masks, boxes = self.detector.predict(image, detection_threshold)
        
        # Step 2: Filter by target categories
        scores_filt, labels_filt, masks_filt, boxes_filt = self.detector.filter_by_categories(
            scores, labels, masks, boxes, categories
        )
        
        # Step 3: Convert masks to binary and create visualization
        mask_threshold = 0.5
        binary_masks = masks_filt > mask_threshold
        self.mask_rcnn_image = self.visualizer.visualize_detections(
            self.image, binary_masks, scores_filt, boxes_filt, labels_filt,
            self.COCO_INSTANCE_CATEGORY_DICT, plot_label=True
        )
        
        # Step 4: Extract geometric features from masks
        self.binary_masks = binary_masks.squeeze(1).numpy()
        self.lower_edges = self.geometry.extract_lower_edges(self.binary_masks)
        self.upper_points = self.geometry.extract_upper_points(self.binary_masks)
        
        # Step 5: Fit edges with RANSAC
        self.ransac_lower_edges = self.geometry.ransac_fit_edges(self.lower_edges, max_trials=100)
        lower_edge_points = self.geometry.get_edge_endpoints(self.ransac_lower_edges)
        
        # Step 6: Reconstruct 3D geometry
        lower_edge_3d_points = self.geometry.reconstruct_3d_edge_points(
            lower_edge_points, self.K, self.T_cam_plane
        )
        lower_3d_points = self.geometry.get_corresponding_3d_point(
            self.lower_edges, self.upper_points, lower_edge_3d_points
        )
        
        # Step 7: Compute object heights
        heights = self.geometry.compute_object_heights(
            lower_3d_points, self.upper_points, self.K, self.T_cam_plane
        )
        
        # Step 8: Construct 3D bounding boxes
        self.bbox_parameters = self.bbox_processor.compute_bbox_parameters(
            lower_edge_3d_points, heights, object_depth=fixed_depth
        )
        
        return self.bbox_parameters
    

    # ==================== Visualization Methods ====================
    
    def get_2d_detected_img(self) -> np.ndarray:
        """
        Get the 2D detection visualization with masks and labels.

        Returns:
            Visualization image with detected objects highlighted
        """
        return self.mask_rcnn_image
    

    def get_lower_edges_img(self, point_size: int = 1) -> np.ndarray:
        """
        Visualize lower edges extracted from binary masks.
        
        Args:
            point_size: Circle radius for edge points
        
        Returns:
            Visualization image with lower edges highlighted
        """
        return self.visualizer.visualize_lower_edges(
            self.binary_masks, self.lower_edges, point_size
        )
    

    def visualize_lower_edges_with_ransac(self, lower_edge_img: np.ndarray,
                                         point_size: int = 1) -> np.ndarray:
        """
        Visualize RANSAC-fitted lower edges.
        
        Args:
            lower_edge_img: Base image (typically from get_lower_edges_img)
            point_size: Circle radius for RANSAC points
        
        Returns:
            Image with RANSAC-fitted edges overlaid
        """
        return self.visualizer.visualize_ransac_edges(
            lower_edge_img, self.ransac_lower_edges, point_size
        )
    

    def visualize_top_points(self, img: np.ndarray, point_size: int = 1) -> np.ndarray:
        """
        Visualize uppermost points of detected objects.
        
        Args:
            img: Base image to draw on
            point_size: Circle radius for points
        
        Returns:
            Image with upper points marked
        """
        return self.visualizer.visualize_points(img, self.upper_points, point_size)
    

    def visualize_bboxes_in_image(self, color: tuple = (0, 255, 0),
                                 line_size: int = 1) -> np.ndarray:
        """
        Visualize 3D bounding boxes projected onto the image.
        
        Args:
            color: Line color in BGR format
            line_size: Line thickness
        
        Returns:
            Image with 3D bounding boxes drawn as wireframes
        """
        bbox_points_3d = self.bbox_processor.bbox_parameters_to_points(self.bbox_parameters)
        bbox_points_2d = self.bbox_processor.project_bboxes_to_image(
            bbox_points_3d, self.K, self.T_cam_plane, self.image.shape
        )
        return self.visualizer.visualize_all_bboxes(
            self.image, bbox_points_2d, color, line_size
        )
    
    def bbox_parameters_to_points(self, bbox_parameters: List[Dict]) -> np.ndarray:
        """
        Convert bounding box parameters to 8 corner points.
        
        Args:
            bbox_parameters: List of bbox parameter dicts
        
        Returns:
            Array of shape (N, 8, 3) with 8 corner points for each bbox
        """
        return self.bbox_processor.bbox_parameters_to_points(bbox_parameters)
    
    
    def visualize_detections(self, img: np.ndarray, binary_masks,
                           scores, boxes, labels, plot_label: bool = False) -> np.ndarray:
        """
        Visualize 2D detections (legacy method for compatibility).
        
        Args:
            img: Input image
            binary_masks: Binary detection masks
            scores: Detection scores
            boxes: Bounding boxes
            labels: Class labels
            plot_label: Whether to draw labels
        
        Returns:
            Visualized image
        """
        return self.visualizer.visualize_detections(
            img, binary_masks, scores, boxes, labels,
            self.COCO_INSTANCE_CATEGORY_DICT, plot_label
        )
