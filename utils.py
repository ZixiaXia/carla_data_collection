import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

# Mapping from category ID to RGB colors for visualization
id_to_color = {
    0: (0, 0, 0),             # Unlabeled - Black
    1: (128, 64, 128),        # Roads - Gray-purple
    2: (244, 35, 232),        # SideWalks - Pink-purple
    3: (70, 70, 70),          # Building - Dark gray
    4: (102, 102, 156),       # Wall - Purple-gray
    5: (190, 153, 153),       # Fence - Light gray-red
    6: (153, 153, 153),       # Pole - Gray
    7: (250, 170, 30),        # TrafficLight - Orange-yellow
    8: (220, 220, 0),         # TrafficSign - Yellow
    9: (107, 142, 35),        # Vegetation - Olive green
    10: (152, 251, 152),      # Terrain - Light green
    11: (70, 130, 180),       # Sky - Sky blue
    12: (220, 20, 60),        # Pedestrian - Red
    13: (255, 0, 0),          # Rider - Dark red
    14: (0, 0, 142),          # Car - Dark blue
    15: (0, 0, 70),           # Truck - Dark gray-blue
    16: (0, 60, 100),         # Bus - Blue-green
    17: (0, 80, 100),         # Train - Dark cyan-blue
    18: (0, 0, 230),          # Motorcycle - Light blue
    19: (119, 11, 32),        # Bicycle - Dark red
    20: (110, 190, 160),      # Static - Light cyan
    21: (170, 120, 50),       # Dynamic - Brown
    22: (55, 90, 80),         # Other - Dark gray-green
    23: (45, 60, 150),        # Water - Blue
    24: (157, 234, 50),       # RoadLine - Light yellow-green
    25: (81, 0, 81),          # Ground - Dark purple
    26: (150, 100, 100),      # Bridge - Light red
    27: (230, 150, 140),      # RailTrack - Pink-orange
    28: (180, 165, 180)       # GuardRail - Light gray-purple
}

# Mapping from category ID to semantic labels
id_to_label = {
    0: 'NavigableArea',  # Roads, Roadlines, Ground, Terrain
    1: 'Sidewalks', 
    2: 'StaticObstacle', 
    3: 'TrafficElement', 
    4: 'EnvironmentalContext', 
}

# Extended mapping for fine-grained semantic categories
id_to_label28 = {
    0: 'Unlabeled',
    1: 'Roads',
    2: 'SideWalks',
    3: 'Building',
    4: 'Wall',
    5: 'Fence',
    6: 'Pole',
    7: 'TrafficLight',
    8: 'TrafficSign',
    9: 'Vegetation',
    10: 'Terrain',
    11: 'Sky',
    12: 'Pedestrian',
    13: 'Rider',
    14: 'Car',
    15: 'Truck',
    16: 'Bus',
    17: 'Train',
    18: 'Motorcycle',
    19: 'Bicycle',
    20: 'Static',
    21: 'Dynamic',
    22: 'Other',
    23: 'Water',
    24: 'RoadLine',
    25: 'Ground',
    26: 'Bridge',
    27: 'RailTrack',
    28: 'GuardRail'
}


def group_classes(image):
    """
    Merge specific categories into broader classes.
    """
    id_to_merged_id = {
        1: [1, 14],  # Roads and Cars
        2: [2],      # Sidewalks
        3: [24],     # Roadlines
    }
    
    merged_array = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for new_id, merged_ids in id_to_merged_id.items():
        for merged_id in merged_ids:
            mask = (image == merged_id)  
            merged_array[mask] = new_id
    
    return merged_array


def map_semantic_colors(image):   
    """
    Map class IDs to their corresponding RGB colors for visualization.
    """
    color_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for category_id, color in id_to_color.items():
        mask = (image == category_id)
        color_image[mask] = color

    return color_image


def dilate_class(matrix, class_id, kernel_size=5, iterations=1):
    """
    Apply dilation to a specific class to expand its region.
    
    Args:
        matrix: 2D numpy array of class labels.
        class_id: The class label to be dilated.
        kernel_size: Size of the structuring element.
        iterations: Number of dilation iterations.
    
    Returns:
        A new matrix with the specified class dilated.
    """
    mask = (matrix == class_id).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    result = matrix.copy()
    result[dilated_mask == 1] = class_id
    
    return result