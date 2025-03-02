import carla
import copy
import math
import numpy as np
from queue import Queue

from config import *
from transform import get_camera_intrinsics
from utils import dilate_class

def create_top_bev_map(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue, location=(0, 0, 8), sensor_name='BEVMAP'):
    IM_WIDTH = CAM_BEV_IM_WIDTH
    IM_HEIGHT = CAM_BEV_IM_HEIGHT
    FOV = CAM_BEV_FOV
    bev_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    bev_bp.set_attribute('image_size_x', "{}".format(IM_WIDTH))
    bev_bp.set_attribute('image_size_y', "{}".format(IM_HEIGHT))
    bev_bp.set_attribute('fov', FOV)  
    
    bev_position = bev_position = carla.Transform(
        carla.Location(x=location[0], y=location[1], z=location[2]), 
        carla.Rotation(pitch=-90, yaw=0, roll=0)
    )
    bev_cam = world.spawn_actor(bev_bp, bev_position, attach_to=ego_vehicle)
    bev_cam.listen(lambda data, actor=bev_cam: sensor_callback(data, sensor_queue, sensor_name, actor))
    sensor_list.append(bev_cam)

    return sensor_list, sensor_queue

def create_bev_view(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue):
    IM_WIDTH = CAM_BEV_IM_HEIGHT
    IM_HEIGHT = CAM_BEV_IM_HEIGHT
    FOV = CAM_BEV_FOV
    
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', "{}".format(IM_WIDTH))
    cam_bp.set_attribute('image_size_y', "{}".format(IM_HEIGHT))
    cam_bp.set_attribute('fov', FOV)  

    camera_transform = carla.Transform(
        carla.Location(x=0.0, z=10.0),  
        carla.Rotation(pitch=-90)  
    )

    camera_sensor = world.spawn_actor(cam_bp, camera_transform, attach_to=ego_vehicle)
    camera_sensor.listen(lambda data: sensor_callback(data, sensor_queue, "BEVVIEW", camera_sensor))
    
    sensor_list.append(camera_sensor)

    return sensor_list, sensor_queue

def create_camera(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue):
    IM_WIDTH = CAM_IM_WIDTH
    IM_HEIGHT = CAM_IM_HEIGHT
    FOV = CAM_FOV
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', "{}".format(IM_WIDTH))
    cam_bp.set_attribute('image_size_y', "{}".format(IM_HEIGHT))
    cam_bp.set_attribute('fov', FOV)  
    camera_positions = {
        'CAM_FRONT': carla.Transform(carla.Location(x=1.0, z=1.5), carla.Rotation(pitch=0, yaw=0, roll=0)),  # front
        'CAM_FRONT_RIGHT': carla.Transform(carla.Location(x=1.0, y=0.5, z=1.5), carla.Rotation(pitch=0, yaw=55, roll=0)),  # front_right
        'CAM_FRONT_LEFT': carla.Transform(carla.Location(x=1.0, y=-0.5, z=1.5), carla.Rotation(pitch=0, yaw=-55, roll=0)),  # front_left
        'CAM_BACK': carla.Transform(carla.Location(x=-1.5, z=2), carla.Rotation(pitch=0, yaw=180, roll=0)),  # back
        'CAM_BACK_LEFT': carla.Transform(carla.Location(y=-1.0, z=1.5), carla.Rotation(pitch=0, yaw=-125, roll=0)),  # back_left
        'CAM_BACK_RIGHT': carla.Transform(carla.Location(y=1.0, z=1.5), carla.Rotation(pitch=0, yaw=125, roll=0)),  # back_right
    }
    for key, position in camera_positions.items():
        if key == 'CAM_BACK':
            cam_bp.set_attribute('fov', CAM_BACK_FOV)  
        cam = world.spawn_actor(cam_bp, position, attach_to=ego_vehicle)
        camera_intrinsics = get_camera_intrinsics(cam, IM_WIDTH, IM_HEIGHT)
        cam.listen(lambda data, key=key, actor=cam, camera_intrinsics=camera_intrinsics: sensor_callback(data, sensor_queue, key, actor, camera_intrinsics))
        sensor_list.append(cam)

    return sensor_list, sensor_queue

def create_lidar(blueprint_library, world, settings, ego_vehicle, sensor_list, sensor_queue):
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')  
    rotation_frequency = 1 / settings.fixed_delta_seconds
    points_per_second = int(1080 * 32 * rotation_frequency)
    lidar_bp.set_attribute('points_per_second', str(int(points_per_second)))  
    lidar_bp.set_attribute('range', '100')  
    lidar_bp.set_attribute('rotation_frequency', str(int(rotation_frequency)))  
    lidar_bp.set_attribute('upper_fov', '10') 
    lidar_bp.set_attribute('lower_fov', '-30')  
    lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5), carla.Rotation(pitch=0, yaw=0, roll=0))  
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
    lidar.listen(lambda data, actor=lidar: sensor_callback(data, sensor_queue, 'LIDAR', actor))
    sensor_list.append(lidar)

    return sensor_list, sensor_queue

def create_radar(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue):
    radar_bp = blueprint_library.find('sensor.other.radar')
    # radar_bp.set_attribute('horizontal_fov', '30') #35
    # radar_bp.set_attribute('vertical_fov', '15') #20
    # radar_bp.set_attribute('range', '250')
    radar_positions = {
        'RADAR_FRONT': carla.Transform(carla.Location(x=2.5, z=1.0), carla.Rotation(pitch=0, yaw=0, roll=0)),  # front
        'RADAR_FRONT_LEFT': carla.Transform(carla.Location(x=1, y=-1.5, z=1.0), carla.Rotation(pitch=0, yaw=-90, roll=0)),  # front left 
        'RADAR_FRONT_RIGHT': carla.Transform(carla.Location(x=1, y=1.5, z=1.0), carla.Rotation(pitch=0, yaw=90, roll=0)),  # front right
        'RADAR_BACK_LEFT': carla.Transform(carla.Location(x=-2.5, y=-1, z=1.0), carla.Rotation(pitch=0, yaw=180, roll=0)),  # back left       
        'RADAR_BACK_RIGHT': carla.Transform(carla.Location(x=-2.5, y=1, z=1.0), carla.Rotation(pitch=0, yaw=180, roll=0)),  # back right
    }
    for key, position in radar_positions.items():
        radar = world.spawn_actor(radar_bp, position, attach_to=ego_vehicle)
        radar.listen(lambda data, key=key, actor=radar: sensor_callback(data, sensor_queue, key, actor))
        sensor_list.append(radar)

    return sensor_list, sensor_queue

def create_imu(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue):
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', '0.05')  
    imu01 = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego_vehicle)
    imu01.listen(lambda data: sensor_callback(data, sensor_queue, 'IMU'))
    sensor_list.append(imu01)

    return sensor_list, sensor_queue

def create_gnss(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue):
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_bp.set_attribute('sensor_tick', '0.05')  
    gnss01 = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=ego_vehicle)
    gnss01.listen(lambda data: sensor_callback(data, sensor_queue, 'GNSS'))
    sensor_list.append(gnss01)

    return sensor_list, sensor_queue

def sensor_callback(data, sensor_queue, sensor_name, actor=None, camera_intrinsics=None):
    transform = actor.get_transform() if actor is not None else None
    sensor_queue.put((data.frame, sensor_name, data, transform, camera_intrinsics))

def setup_sensors(blueprint_library, world, settings, ego_vehicle, sensor_list, sensor_queue):
    sensor_list, sensor_queue = create_camera(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue)
    sensor_list, sensor_queue = create_bev_view(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue)
    sensor_list, sensor_queue = create_lidar(blueprint_library, world, settings, ego_vehicle, sensor_list, sensor_queue)
    sensor_list, sensor_queue = create_radar(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue)
    sensor_list, sensor_queue = create_imu(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue)
    sensor_list, sensor_queue = create_gnss(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue)
    sensor_list, sensor_queue = create_top_bev_map(blueprint_library, world, ego_vehicle, sensor_list, sensor_queue)
    
    return sensor_queue, sensor_list

def _parse_cam_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def _parse_bev_map_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, 2]  

    array = dilate_class(array, class_id=24)

    return array

def _parse_lidar_cb(lidar_data, ego_vehicle, transform, visualize=False):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = copy.deepcopy(points)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    points[:, [0, 1]] = points[:, [1, 0]]
    
    if visualize:
        current_rot = lidar_data.transform.rotation
        for point in points:
            x, y, z, intensity = point 
            fw_vec = carla.Vector3D(x=float(x), y=float(y), z=float(z))
            transform = carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch,
                    yaw=current_rot.yaw,
                    roll=current_rot.roll
                )
            )
            transform.transform(fw_vec)

            world = ego_vehicle.get_world()
            world.debug.draw_point(
                location=lidar_data.transform.location + fw_vec,
                size=0.1,  
                color=carla.Color(0, 255, 0),  
                life_time=0.1 
            )
    return points

def _parse_radar_cb(radar_data, ego_vehicle, sensor_idx, visualize=False):
    points = [] 

    ego_velocity = ego_vehicle.get_velocity()
    current_rot = radar_data.transform.rotation
    timestamp = radar_data.timestamp
    for detection in radar_data:
        azi = math.degrees(detection.azimuth)
        alt = math.degrees(detection.altitude)
        # print(f"Depth: {detection.depth}, Azimuth: {azi}, Altitude: {alt}")

        fw_vec = carla.Vector3D(
            x=detection.depth * math.cos(detection.azimuth) * math.cos(detection.altitude),
            y=detection.depth * math.sin(detection.azimuth) * math.cos(detection.altitude),
            z=detection.depth * math.sin(detection.altitude)
        )
        # fw_vec = carla.Vector3D(x=detection.depth - 0.25)
        x, y, z = fw_vec.x, fw_vec.y, fw_vec.z
        # if "FRONT_LEFT" == sensor_idx:
        #     print("x, y, z:", x, y, z)

        vx = detection.velocity * np.cos(detection.azimuth)
        vy = detection.velocity * np.sin(detection.azimuth)
        vx_comp = vx - ego_velocity.x
        vy_comp = vy - ego_velocity.y

        speed = math.sqrt(vx_comp**2 + vy_comp**2)
        dyn_prop = 1 if speed >= 0.1 else 0 

        point = [
            x, y, z, dyn_prop,  
            azi, alt, detection.depth, detection.velocity,  
            vx, vy, vx_comp, vy_comp,  
        ]
        points.append(point)

        if visualize:
            transform = carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll
                )
            )
            transform.transform(fw_vec)
            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            velocity_range = 7.5
            norm_velocity = detection.velocity / velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(-1.0, 0.0, -1.0 - norm_velocity)) * 255.0)
            world = ego_vehicle.get_world()
            world.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))
    
    points = np.array(points)
    # points = np.reshape(points, (-1, ))
    return points

def _parse_imu_cb(imu_data, current_time, pose_estimator=None):
    # pose = pose_estimator.update(imu_data, current_time)

    linear_acceleration = np.array([
        imu_data.accelerometer.x,
        imu_data.accelerometer.y,
        imu_data.accelerometer.z
    ])
    
    angular_velocity = np.array([
        imu_data.gyroscope.x,
        imu_data.gyroscope.y,
        imu_data.gyroscope.z
    ])

    # print(f"Linear Acceleration (m/s²): x={linear_acceleration[0]:.2f}, y={linear_acceleration[1]:.2f}, z={linear_acceleration[2]:.2f}")
    # print(f"Angular Velocity (rad/s):   x={angular_velocity[0]:.2f}, y={angular_velocity[1]:.2f}, z={angular_velocity[2]:.2f}")

    imu_parsed = {
        'linear_acceleration': linear_acceleration,
        'angular_velocity': angular_velocity,
        'timestamp': current_time
    }

    return imu_parsed

def _parse_gnss_cb(s_data):
    gnss_data = {
        'latitude': s_data.latitude,
        'longitude': s_data.longitude,
        'altitude': s_data.altitude
    }

    return gnss_data

def _parse_rgb_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array