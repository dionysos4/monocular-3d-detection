from monocular3d import MonoDetection
import numpy as np
import matplotlib.pyplot as plt


# transformation to project from plane to camera coordinates
T_cam_plane = np.array([[ 9.99980405e-01,  6.26016000e-03,  0.00000000e+00,
            2.23832138e-02],
        [-6.18763142e-03,  9.88394890e-01, -1.51780274e-01,
            3.53400778e+00],
        [-9.50168804e-04,  1.51777300e-01,  9.88414259e-01,
            5.42680020e-01],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            1.00000000e+00]])

# camera intrinsic matrix
K = np.array([[2.19132030e+03, 0.00000000e+00, 1.00070519e+03],
       [0.00000000e+00, 2.19157698e+03, 5.76974055e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

img_left = plt.imread("imgs/input_img.jpg") / 255.0
img_left = (img_left).astype(np.float32)

mono_detection = MonoDetection()
bbox_params = mono_detection.predict(img_left, K, T_cam_plane, 
                       fixed_depth=3.66, categories=["boat"], 
                       detection_threshold=0.7)

print("Predicted bounding box parameters: \n", bbox_params)

# Visualize the results

# mask rcnn detections
plt.figure(1)
plt.imshow(mono_detection.get_2d_detected_img())
plt.show()

# lower edge detection
lower_edge_img = mono_detection.get_lower_edges_img(point_size=2)
plt.figure(2)
plt.imshow(lower_edge_img)
plt.show()

# ransac line detection
ransac_lower_edges_img = mono_detection.visualize_lower_edges_with_ransac(lower_edge_img, point_size=2)
plt.figure(3)
plt.imshow(ransac_lower_edges_img)
plt.show()

# top point detection
top_point_ransac_img = mono_detection.visualize_top_points(ransac_lower_edges_img, point_size=5)
plt.figure(4)
plt.imshow(top_point_ransac_img)
plt.show()

# 3d detection visualization
plt.figure(5)
plt.imshow(mono_detection.visualize_bboxes_in_image(color=(228/255, 26/255, 26/255), line_size=2))
plt.show()