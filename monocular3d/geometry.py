import numpy as np
from sklearn.linear_model import RANSACRegressor
from typing import List, Tuple


class GeometryProcessor:
    """
    Processes geometric computations for 3D reconstruction from 2D detections.
    
    Handles:
    - Edge extraction from binary masks
    - RANSAC regression on edges
    - 3D point reconstruction from 2D points
    - Height estimation
    """
    
    def extract_lower_edges(self, binary_masks: np.ndarray) -> List[np.ndarray]:
        """
        Extract lower edge points from binary masks.
        
        Args:
            binary_masks: Binary masks of shape (N, H, W)
            
        Returns:
            List of lower edge coordinates for each mask
        """
        lower_edges = []
        for i in range(binary_masks.shape[0]):
            H, W = binary_masks[i].shape
            row_indices = np.arange(H)[:, None]
            masked_rows = np.where(binary_masks[i], row_indices, -1)
            lower_edge_y = masked_rows.max(axis=0)
            x_coords = np.where(lower_edge_y >= 0)[0]
            lower_edge_coords = np.stack([x_coords, lower_edge_y[x_coords]], axis=1)
            lower_edges.append(lower_edge_coords)
        return lower_edges
    

    def extract_upper_points(self, binary_masks: np.ndarray) -> np.ndarray:
        """
        Extract uppermost point from binary masks.
        
        Args:
            binary_masks: Binary masks of shape (N, H, W)
            
        Returns:
            Array of upper points of shape (N, 2)
        """
        upper_points = []
        for i in range(binary_masks.shape[0]):
            H, W = binary_masks[i].shape
            row_indices = np.arange(H)[:, None]
            masked_rows = np.where(binary_masks[i], row_indices, H+1)
            upper_edge_y = masked_rows.min()
            x_coords = np.where(masked_rows == upper_edge_y)[1]
            x_coord = np.mean(x_coords).astype(int)
            upper_points.append(np.array([x_coord, upper_edge_y]))
        return np.array(upper_points)
    

    def ransac_fit_edge(self, edge: np.ndarray, max_trials: int = 100) -> Tuple:
        """
        Perform RANSAC regression on a single edge.
        
        Args:
            edge: Edge coordinates of shape (N, 2)
            max_trials: Maximum RANSAC iterations
            
        Returns:
            Tuple of (X, y_pred) or (None, None) if insufficient points
        """
        if edge.shape[0] < 2:
            return None, None
        
        X = edge[:, 0].reshape(-1, 1)
        y = edge[:, 1]
        model = RANSACRegressor(max_trials=max_trials)
        model.fit(X, y)
        y_pred = model.predict(X)
        return X.flatten(), y_pred
    

    def ransac_fit_edges(self, edges: List[np.ndarray], max_trials: int = 100) -> List[Tuple]:
        """
        Apply RANSAC regression to multiple edges.
        
        Args:
            edges: List of edge coordinates
            max_trials: Maximum RANSAC iterations
            
        Returns:
            List of (X, y_pred) tuples
        """
        ransac_edges = []
        for edge in edges:
            X_ransac, y_ransac_pred = self.ransac_fit_edge(edge, max_trials=max_trials)
            if X_ransac is None:
                print("Warning: Not enough points for RANSAC regression.")
            ransac_edges.append((X_ransac, y_ransac_pred))
        return ransac_edges
    

    def get_edge_endpoints(self, ransac_edges: List[Tuple]) -> np.ndarray:
        """
        Extract start and end points from RANSAC-fitted edges.
        
        Args:
            ransac_edges: List of (X, y_pred) tuples from RANSAC
            
        Returns:
            Array of shape (N, 4) with (x_start, y_start, x_end, y_end)
        """
        edge_points = []
        for x, y in ransac_edges:
            if x is not None and y is not None:
                edge_points.append([x[0], y[0], x[-1], y[-1]])
        return np.array(edge_points)
    

    def compute_reconstruction_matrices(self, K: np.ndarray, T_cam_plane: np.ndarray) -> Tuple:
        """
        Compute reconstruction matrices for 3D point calculation.
        
        Args:
            K: Camera intrinsic matrix (3, 3)
            T_cam_plane: Transformation from plane to camera (4, 4)
            
        Returns:
            Tuple of (rec_mat, mulvec) for reconstruction
        """
        R_cam_plane = T_cam_plane[:3, :3]
        t_cam_plane = T_cam_plane[:3, 3]
        P = K @ R_cam_plane
        
        # Plane equation coefficients: ax + by + cz + d = 0
        # For horizontal plane: y = 0 -> 0x - 1y + 0z + 0 = 0
        a, b, c, d = 0, -1, 0, 0
        
        rec_mat = np.zeros((4, 4))
        rec_mat[:3, :3] = P
        rec_mat[3, :3] = np.array([a, b, c])
        rec_mat[3, 3] = d
        
        mulvec = np.zeros(4)
        mulvec[:3] = -K @ t_cam_plane
        mulvec[3] = -d
        
        return rec_mat, mulvec
    

    def reconstruct_3d_edge_points(self, edge_points_2d: np.ndarray, 
                                   rec_mat: np.ndarray, mulvec: np.ndarray) -> np.ndarray:
        """
        Reconstruct 3D points from 2D edge endpoints.
        
        Args:
            edge_points_2d: 2D edge endpoints (N, 4) with (x_start, y_start, x_end, y_end)
            rec_mat: Reconstruction matrix (4, 4)
            mulvec: Multiplication vector (4,)
            
        Returns:
            3D points of shape (N, 2, 4) with homogeneous coordinates
        """
        points_3d = []
        rec_mat = rec_mat.copy()  # Don't modify original
        
        for edge_points in edge_points_2d:
            # Reconstruct start point
            rec_mat[0, 3] = -edge_points[0]
            rec_mat[1, 3] = -edge_points[1]
            rec_mat[2, 3] = -1
            p0 = np.linalg.inv(rec_mat) @ mulvec
            p0[-1] = 1
            
            # Reconstruct end point
            rec_mat[0, 3] = -edge_points[2]
            rec_mat[1, 3] = -edge_points[3]
            p1 = np.linalg.inv(rec_mat) @ mulvec
            p1[-1] = 1
            
            points_3d.append((p0, p1))
        
        return np.array(points_3d)
    

    def get_corresponding_3d_point(self, lower_edges: List[np.ndarray], 
                                   upper_points: np.ndarray,
                                   lower_edge_3d_points: np.ndarray) -> np.ndarray:
        """
        Find 3D point on lower edge corresponding to upper point.
        
        Args:
            lower_edges: List of lower edge coordinates
            upper_points: Upper point coordinates (N, 2)
            lower_edge_3d_points: 3D edge points (N, 2, 4)
            
        Returns:
            Corresponding 3D points (N, 4)
        """
        mid_3d_points = []
        for lower_edge, upper_point, edge_3d in zip(lower_edges, upper_points, lower_edge_3d_points):
            idx_in_edge = np.where(lower_edge[:, 0] == upper_point[0])[0]
            t = idx_in_edge / len(lower_edge)
            mid_3d_point = edge_3d[0] + t * (edge_3d[1] - edge_3d[0])
            mid_3d_points.append(mid_3d_point)
        return np.array(mid_3d_points)
    
    
    def compute_object_heights(self, lower_3d_points: np.ndarray, 
                              upper_points: np.ndarray,
                              K: np.ndarray, T_cam_plane: np.ndarray) -> np.ndarray:
        """
        Compute object heights from upper and lower points.
        
        Args:
            lower_3d_points: 3D points at object base (N, 4)
            upper_points: 2D upper points in image (N, 2)
            K: Camera intrinsic matrix (3, 3)
            T_cam_plane: Transformation matrix (4, 4)
            
        Returns:
            Object heights (N,)
        """
        heights = []
        T_plane_cam = np.linalg.inv(T_cam_plane)
        camera_height = T_plane_cam[1, 3]
        
        for upper_point, mid_3d_point in zip(upper_points, lower_3d_points):
            # Convert upper point to plane coordinates
            point_img_hom = np.array([
                upper_point[0] - K[0, 2],
                upper_point[1] - K[1, 2],
                K[0, 0], 1
            ])
            point_plane_hom = T_plane_cam @ point_img_hom
            
            # Compute height using trigonometry
            alpha = np.arctan(
                (camera_height - point_plane_hom[1]) / 
                np.sqrt(point_plane_hom[0]**2 + point_plane_hom[2]**2)
            )
            object_distance = np.linalg.norm(mid_3d_point[:3])
            height = np.tan(alpha) * object_distance - camera_height
            heights.append(height)
        
        return np.array(heights)
