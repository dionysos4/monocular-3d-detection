import numpy as np
from typing import List, Dict


class BBoxProcessor:
    """
    Constructs and transforms 3D bounding boxes.
    
    Handles:
    - Computing bbox parameters from 3D edges and heights
    - Converting bbox parameters to 3D corner points
    - Projecting 3D bboxes to image space
    """
    
    def compute_bbox_parameters(self, lower_edge_3d_points: np.ndarray,
                               object_heights: np.ndarray,
                               object_depth: float = 3.0) -> List[Dict]:
        """
        Compute 3D bounding box parameters.
        
        Args:
            lower_edge_3d_points: 3D edge points (N, 2, 4)
            object_heights: Object heights (N,)
            object_depth: Fixed depth for all objects
            
        Returns:
            List of bbox parameter dicts with keys: x, y, z, length, height, width, yaw
        """
        bbox_parameters = []
        
        for edge_points, height in zip(lower_edge_3d_points, object_heights):
            p0 = edge_points[0]
            p1 = edge_points[1]
            
            # Ensure consistent ordering
            if p0[0] > p1[0]:
                p0, p1 = p1, p0
            
            # Compute orthogonal direction for depth
            normalized_vector = (p1[:3] - p0[:3]) / np.linalg.norm(p1[:3] - p0[:3])
            orthogonal_vector = np.cross(normalized_vector, np.array([0, 1, 0]))
            orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)
            
            # Check direction towards origin
            vector_to_origin = -p0[:3]
            if np.dot(orthogonal_vector, vector_to_origin) > 1:
                orthogonal_vector = -orthogonal_vector
            
            # Compute 4 base corners
            p2 = p0[:3] + orthogonal_vector * object_depth
            p3 = p1[:3] + orthogonal_vector * object_depth
            
            # Reorder corners: 0---1
            #                   |   |
            #                   3---2
            p0, p1, p2, p3 = p0[:3], p2, p3, p1[:3]
            
            # Compute bbox center and dimensions
            center = (p0 + p1 + p2 + p3) / 4
            o_length = np.linalg.norm(p0 - p3)
            o_width = object_depth
            o_height = height
            
            # Compute yaw angle
            p2_c = p2 - center
            p3_c = p3 - center
            x_rot = p2_c - p3_c
            x_rot = x_rot / np.linalg.norm(x_rot)
            yaw = np.arctan2(x_rot[2], x_rot[0])
            
            bbox_dict = {
                "x": center[0],
                "y": center[1],
                "z": center[2],
                "length": o_length,
                "height": o_height,
                "width": o_width,
                "yaw": yaw
            }
            bbox_parameters.append(bbox_dict)
        
        return bbox_parameters
    
    
    def bbox_parameters_to_points(self, bbox_parameters: List[Dict]) -> np.ndarray:
        """
        Convert bbox parameters to 8 corner points.
        
        Args:
            bbox_parameters: List of bbox dicts with x, y, z, length, height, width, yaw
            
        Returns:
            Array of shape (N, 8, 3) with 8 corner points for each bbox
        """
        bbox_points = []
        
        for bbox in bbox_parameters:
            x = bbox["x"]
            y = bbox["y"]
            z = bbox["z"]
            length = bbox["length"]
            height = bbox["height"]
            width = bbox["width"]
            yaw = bbox["yaw"]
            
            # Rotation matrix around y-axis (pointing downwards)
            R_yaw = np.array([
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)]
            ]).T
            
            # 8 corner points in local coordinate system
            # Bottom face (y=0), then top face (y=-height)
            corners = np.array([
                [-width/2, 0, length/2],
                [width/2, 0, length/2],
                [width/2, 0, -length/2],
                [-width/2, 0, -length/2],
                [-width/2, -height, length/2],
                [width/2, -height, length/2],
                [width/2, -height, -length/2],
                [-width/2, -height, -length/2]
            ])
            
            # Transform to global coordinates
            rotated_corners = (R_yaw @ corners.T).T
            translated_corners = rotated_corners + np.array([x, y, z])
            bbox_points.append(translated_corners)
        
        return np.array(bbox_points)
    
    
    def project_bboxes_to_image(self, bbox_points: np.ndarray, 
                                K: np.ndarray, T_cam_plane: np.ndarray,
                                image_shape: tuple) -> np.ndarray:
        """
        Project 3D bounding boxes to image coordinates.
        
        Args:
            bbox_points: 3D bbox corner points (N, 8, 3)
            K: Camera intrinsic matrix (3, 3)
            T_cam_plane: Transformation from plane to camera (4, 4)
            image_shape: Image dimensions (height, width)
            
        Returns:
            2D bbox points (N, 8, 2) clipped to image bounds
        """
        h, w = image_shape[:2]
        
        # Convert to homogeneous coordinates
        bbox_points_hom = np.concatenate(
            (bbox_points, np.ones((bbox_points.shape[0], bbox_points.shape[1], 1))),
            axis=2
        )
        
        # Transform to camera coordinates
        bbox_points_cam = np.einsum('ij,nkj->nki', T_cam_plane, bbox_points_hom)
        
        # Project to image plane
        bbox_points_img_hom = np.einsum('ij,nkj->nki', K, bbox_points_cam[:, :, :3])
        bbox_points_img = bbox_points_img_hom[:, :, :2] / bbox_points_img_hom[:, :, 2:3]
        
        # Clip to image boundaries
        bbox_points_img[:, :, 0] = np.clip(bbox_points_img[:, :, 0], 0, w-1)
        bbox_points_img[:, :, 1] = np.clip(bbox_points_img[:, :, 1], 0, h-1)
        
        return bbox_points_img
