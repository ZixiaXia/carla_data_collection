import numpy as np
import os
import os.path as osp
import cv2
import math
import json
from PIL import Image

from config import args, carla_palette, CAM_BEV_IM_HEIGHT, CAM_BEV_IM_WIDTH
from config import *
from transform import get_transform_matrix, compute_transform, get_camera_intrinsics
from utils import map_semantic_colors, group_classes

def save_pcd_file(filename, points):
    points_np = np.array(points).astype(np.float32)
    num_points = points_np.shape[0]
    header = f"""# .PCD v0.7 - Point Cloud Data file format
            VERSION 0.7
            FIELDS x y z dyn_prop azi alt depth velocity vx vy vx_comp vy_comp
            SIZE 4 4 4 4 4 4 4 4 4 4 4 4
            TYPE F F F F F F F F F F F F
            COUNT 1 1 1 1 1 1 1 1 1 1 1 1
            WIDTH {num_points}
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS {num_points}
            DATA binary
            """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, points_np, fmt="%.6f %.6f %.6f %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f")

def _parse_data_dict(info):
    # (Pdb) info.keys()
    # dict_keys(['weather', 'timestamp', 'ego2global', 'lidar2ego', 'cams', 'bevseg_path', 'radars', 'lidar_path', 'lidar2global', 'sweeps'])

    data = dict(
        lidar_data=info["lidar_data"],
        sweeps=info["sweeps"],
        timestamp=info["timestamp"],
        radar=info['radars'],
        ego2global=info["ego2global"],
        lidar2ego=np.array(info["lidar2ego"]),
        lidar2global=info['lidar2global'],
        bevseg_map=info["bev_map"],
    )

    data["img"] = []
    data["lidar2camera"] = []
    data["lidar2image"] = []
    data["camera2ego"] = []
    data["camera_intrinsics"] = []
    data["camera2lidar"] = []

    for _, camera_info in info["cams"].items():
        data["img"].append(camera_info["img"])
        data["lidar2camera"].append(camera_info["lidar2camera"])
        data["camera_intrinsics"].append(camera_info["camera_intrinsics"])
        data["lidar2image"].append(camera_info["lidar2image"])
        data["camera2ego"].append(camera_info["camera2ego"])
        data["camera2lidar"].append(camera_info["camera2lidar"])

    return data

def dilate_semantic_map(semantic_map, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_map = cv2.dilate(semantic_map, kernel, iterations=iterations)

    return dilated_map

def save_data(cam_data, lidar_data, radar_data, bev_map, w_frame,  
                transform_data, camera_intrinsics_data, lidar_deque, 
                radar_deque, timestamp, ego_vehicle, weather, 
                bev_view, imu_data, gnss_data, segment_folder=None):
    if segment_folder is None:
        segment_folder = data_root
    data_dict = {} 
    data_dict['weather'] = weather
    data_dict['timestamp'] = timestamp
    data_dict['ego2global'] = get_transform_matrix(ego_vehicle.get_transform())
    data_dict['lidar2ego'] = compute_transform(transform_data['LIDAR'], ego_vehicle.get_transform())
    file_name = f"{w_frame}"

    file_path = osp.join(segment_folder, "BEV_MAP", f"{file_name}.npy")
    np.save(file_path, bev_map)
    data_dict['bev_map'] = file_path
    bev_map_color = map_semantic_colors(group_classes(bev_map))
    cv2.imwrite(osp.join(segment_folder, "BEV_MAP_COLOR", f"{file_name}.png"), bev_map_color)
    cv2.imwrite(osp.join(segment_folder, "BEV_VIEW", f"{file_name}.png"), bev_view)

    data_dict['cams'] = {}
    images = []
    for cam_location in cam_data.keys():       
        img = np.array(cam_data[cam_location])
        # img = cv2.resize(img, (SAVE_CAM_IM_WIDTH, SAVE_CAM_IM_HEIGHT))
        file_path = osp.join(segment_folder, cam_location, f"{file_name}.png")
        cv2.imwrite(file_path, img)
        cv2.putText(img, cam_location, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        images.append(img)

        data_dict['cams'][cam_location] = {}
        data_dict['cams'][cam_location]['data_path'] = file_path
        data_dict['cams'][cam_location]['lidar2camera'] = compute_transform(transform_data['LIDAR'], transform_data[cam_location])
        data_dict['cams'][cam_location]['camera_intrinsics'] = camera_intrinsics_data[cam_location]
        data_dict['cams'][cam_location]['lidar2image'] = np.dot(data_dict['cams'][cam_location]['camera_intrinsics'],  # camera to image
                                                                data_dict['cams'][cam_location]['lidar2camera'])       # lidar to camera
        data_dict['cams'][cam_location]['camera2ego'] = compute_transform(transform_data[cam_location], ego_vehicle.get_transform())
        data_dict['cams'][cam_location]['camera2lidar'] = compute_transform(transform_data[cam_location], transform_data['LIDAR'])

    data_dict['radars'] = {}
    for radar_location in radar_data.keys():        
        single_radar_dict = {}
        file_path = osp.join(segment_folder, radar_location, f"{file_name}.pcd")
        save_pcd_file(file_path, radar_data[radar_location])
        single_radar_dict['radar_data'] = file_path
        single_radar_dict['timestamp'] = timestamp
        single_radar_dict['radar2global'] = get_transform_matrix(transform_data[radar_location])

        radar_deque[radar_location].appendleft(single_radar_dict)
        data_dict['radars'][radar_location]=list(radar_deque[radar_location]) 

    file_path = osp.join(segment_folder, "LIDAR_TOP", f"{file_name}.npy")
    np.save(file_path, lidar_data.astype(np.float32))
    data_dict['lidar_path'] = file_path
    data_dict['lidar2global'] = get_transform_matrix(transform_data['LIDAR'])
    
    data_dict['sweeps'] = list(lidar_deque)
    single_lidar_dict = {}
    single_lidar_dict['data_path'] = file_path
    single_lidar_dict['timestamp'] = timestamp
    single_lidar_dict['lidar2global'] = get_transform_matrix(transform_data['LIDAR'])
    lidar_deque.appendleft(single_lidar_dict)

    file_path = osp.join(segment_folder, "IMU", f"{file_name}.npy")
    np.save(file_path, imu_data)

    file_path = osp.join(segment_folder, "GNSS", f"{file_name}.npy")
    np.save(file_path, gnss_data)

    vehicle_transform = ego_vehicle.get_transform()
    vehicle_transform = {
        'location': {
            'x': vehicle_transform.location.x,
            'y': vehicle_transform.location.y,
            'z': vehicle_transform.location.z
        },
        'rotation': {
            'pitch': vehicle_transform.rotation.pitch,
            'yaw': vehicle_transform.rotation.yaw,
            'roll': vehicle_transform.rotation.roll
        }
    }
    file_path = osp.join(segment_folder, "VEHICLE_TRANSFORM", f"{file_name}.npy")
    np.save(file_path, vehicle_transform)

    # return _parse_data_dict(data_dict)
    return data_dict