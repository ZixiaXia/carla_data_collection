import os
import carla
import random
import numpy as np
import cv2
import time
import json
from threading import Thread
from queue import Queue, Empty
from collections import deque
from tqdm import tqdm


from data import save_data
from sensors import setup_sensors, _parse_cam_cb, _parse_bev_map_cb, _parse_lidar_cb, _parse_radar_cb, _parse_imu_cb, _parse_gnss_cb, _parse_rgb_cb
from sensors import *
from config import args, carla_mkdir_folder
from config import *


def carla_main_loop():
    carla_mkdir_folder(data_root)
    
    actor_list, sensor_list = [], []

    client = carla.Client('localhost', 2000)  # Connect to the simulator
    client.set_timeout(10.0)
    world = client.load_world(carla_map)  # Load Town03 map
    blueprint_library = world.get_blueprint_library()

    try:
        world.set_weather(weather_type)

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        spawn_point = spawn_points[0]
        ego_vehicle_bp = random.choice(blueprint_library.filter("model3"))
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)

        tm = client.get_trafficmanager(8010)
        tm.set_synchronous_mode(True)
        tm.global_percentage_speed_difference(10.0)

        ego_vehicle.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(ego_vehicle, 100)
        tm.ignore_signs_percentage(ego_vehicle, 100)   

        sensor_queue = Queue()
        sensor_queue, sensor_list = setup_sensors(blueprint_library, world, settings, ego_vehicle, sensor_list, sensor_queue)

        frame_idx = -1 
        stop_point = 0

        file_dict = {"infos": []} 
        lidar_deque = deque(maxlen=10)  
        radar_deque = {}
        for radar_location in args.radar_locations:
            radar_deque[radar_location] = deque(maxlen=10) 

        for frame_idx in tqdm(range(num_frame+20)):
            world.tick()

            loc = ego_vehicle.get_transform().location
            spectator.set_transform(carla.Transform(carla.Location(x=loc.x,y=loc.y,z=35),carla.Rotation(yaw=0,pitch=-90,roll=0)))

            snapshot = world.get_snapshot()
            timestamp = snapshot.timestamp.elapsed_seconds
            w_frame = snapshot.frame 

            try:
                cam_data, radar_data, transform_data, camera_intrinsics_data = {}, {}, {}, {}
                for i in range (0, len(sensor_list)):
                    s_frame, s_name, s_data, s_transform, camera_intrinsics = sensor_queue.get(True, 1.0)
                    # print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == 'CAM':
                        sensor_idx = s_name
                        cam_data[sensor_idx] = _parse_cam_cb(s_data)
                        transform_data[sensor_idx] = s_transform
                        camera_intrinsics_data[sensor_idx] = camera_intrinsics
                        # print(sensor_idx, s_transform)
                    elif sensor_type == 'BEVMAP':
                        bev_map = _parse_bev_map_cb(s_data)
                        transform_data['BEVMAP'] = s_transform
                    elif sensor_type == 'LIDAR':
                        lidar_data = _parse_lidar_cb(s_data, ego_vehicle, s_transform)
                        transform_data['LIDAR'] = s_transform
                        # print('LIDAR', s_transform)
                    elif sensor_type == 'RADAR':
                        sensor_idx = s_name
                        radar_data[sensor_idx] = _parse_radar_cb(s_data, ego_vehicle, sensor_idx)
                        transform_data[sensor_idx] = s_transform
                        # print(sensor_idx, s_transform)
                    elif sensor_type == 'IMU':
                        imu_data = _parse_imu_cb(s_data, timestamp)
                        imu_raw_data = s_data
                    elif sensor_type == 'GNSS':
                        gnss_data = _parse_gnss_cb(s_data)
                    elif s_name == 'BEVVIEW':
                        bev_view = _parse_rgb_cb(s_data)

                if frame_idx < 20:
                    continue # the very start point is unstable
                
                velocity = ego_vehicle.get_velocity()
                speed = int(3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))
                if not speed > 0:
                    print(f"Frame: {frame_idx}, Speed: {np.round(speed)} km/h, Vehicle Stopped!!!")
                    stop_point += 1
                if stop_point > 10:
                    print("\033[91mThe Vehicle stopped too long!\033[0m")
                    break
                    
                data_dict = save_data(cam_data, lidar_data, radar_data, bev_map, w_frame, 
                                        transform_data, camera_intrinsics_data, lidar_deque, radar_deque, timestamp, ego_vehicle, weather, bev_view, imu_data, gnss_data)    
                file_dict["infos"].append(data_dict)

            except Empty:
                print("     Something wrong.")
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  
            raise TypeError(f"unserializabel type: {type(obj)}")

        with open(os.path.join(data_root, 'data.json'), 'w') as json_file:
            json.dump(file_dict, json_file, indent=4, default=convert_to_serializable)
            print(f"\033[92m=====FINISH DATA COLLECTION: {frame_idx} FRAMES!=====\033[0m")

    finally:
        if 'rgb_camera' in locals():
            rgb_camera.stop()
        if 'depth_camera' in locals():
            depth_camera.stop()
        if 'ego_vehicle' in locals():
            ego_vehicle.destroy()
        world.apply_settings(original_settings)

if __name__ == '__main__':
    carla_main_loop()

    try:
        carla_main_loop()
    except KeyboardInterrupt:
        print("Exiting program.")
