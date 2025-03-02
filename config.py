import argparse
import os
import shutil
import carla

def get_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='CARLA Data Collection Script')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server IP')
    parser.add_argument('--port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--tm_port', default=8000, type=int, help='Traffic Manager port')
    parser.add_argument('--cam-locations', default=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'], help='Camera locations')
    parser.add_argument('--radar-locations', default=['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'], help='Radar locations')
    parser.add_argument('--save-path', default='carla/', help='Path to save collected data')
    return parser.parse_args()

def carla_mkdir_folder(
    path, 
    sensor_types=['CAM', 'LIDAR_TOP', 'IMU', 'GNSS', 'RADAR', 'BEV_MAP', 'BEV_MAP_COLOR', 'BEV_VIEW', 'VEHICLE_TRANSFORM'],
    cam_locations=['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT'], 
    radar_locations=['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK_LEFT', 'BACK_RIGHT']
):
    """
    Create directories for CARLA data storage. If the path exists, delete all existing files and folders first.
    """
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):  # Remove files or symbolic links
                    os.remove(file_path)
                elif os.path.isdir(file_path):  # Remove subdirectories and their contents
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"Creating new directory: {path}")

    for s_type in sensor_types:
        if s_type == 'CAM':
            for location in cam_locations:
                os.makedirs(os.path.join(path, f"CAM_{location}"), exist_ok=True)
        elif s_type == 'RADAR':
            for location in radar_locations:
                os.makedirs(os.path.join(path, f"RADAR_{location}"), exist_ok=True)
        else:
            os.makedirs(os.path.join(path, s_type), exist_ok=True)

# Get command-line arguments
args = get_args()

# CARLA semantic segmentation color palette
carla_palette = {
    0: (0, 0, 0),          # None (unlabeled area, black)
    1: (70, 70, 70),       # Roads (dark gray)
    2: (100, 40, 40),      # Sidewalks (dark red)
    3: (55, 90, 80),       # Buildings (gray-green)
    4: (220, 20, 60),      # Walls (red)
    5: (153, 153, 153),    # Fences (light gray)
    6: (157, 234, 50),     # Poles (light yellow-green)
    7: (128, 64, 128),     # TrafficLight (purple)
    8: (244, 35, 232),     # TrafficSigns (pink-purple)
    9: (107, 142, 35),     # Vegetation (olive green)
    10: (0, 0, 142),       # Terrain (dark blue)
    11: (102, 102, 156),   # Sky (purple-gray)
    12: (220, 20, 60),     # Pedestrians (red)
    13: (255, 0, 0),       # Rider (deep red)
    14: (0, 0, 142),       # Car (dark blue)
    15: (0, 0, 70),        # Truck (dark gray-blue)
    16: (0, 60, 100),      # Bus (blue-green)
    17: (0, 80, 100),      # Train (dark cyan-blue)
    18: (0, 0, 230),       # Motorcycle (light blue)
    19: (119, 11, 32),     # Bicycle (dark red)
    20: (250, 170, 30),    # Static (orange-yellow)
    21: (110, 190, 160),   # Dynamic (light green)
    22: (170, 120, 50),    # Other (brown)
    23: (0, 130, 180),     # Water (aqua blue)
}

# Camera and rendering settings
CAM_BEV_IM_WIDTH = 500
CAM_BEV_IM_HEIGHT = 500
CAM_IM_WIDTH = 800
CAM_IM_HEIGHT = 450

CAM_FOV = '70'
CAM_BACK_FOV = '110'
CAM_BEV_FOV = '120'

# Weather conditions
weather = 'Sunny'
weather_dict = {
    'Sunny': carla.WeatherParameters.ClearNoon,
    'Rainy': carla.WeatherParameters.MidRainyNoon,
    'Night': carla.WeatherParameters.ClearNight,
}
weather_type = weather_dict[weather]

# Simulation parameters
n_segments = 2
vehicle_num = 0

# CARLA map and dataset settings
carla_map = 'Town01'
num_frame = 400
test_mode = False

data_root = f"data/carla_1/train/{carla_map}_{weather}_{vehicle_num}_vehicles"