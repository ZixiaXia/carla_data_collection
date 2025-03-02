import carla
import numpy as np

def get_transform_matrix(transform):
    """Convert CARLA transform (location and rotation) to a 4x4 transformation matrix."""
    rotation = transform.rotation
    location = transform.location
    
    # Convert yaw, pitch, roll to radians
    roll = np.deg2rad(rotation.roll)
    pitch = np.deg2rad(rotation.pitch)
    yaw = np.deg2rad(rotation.yaw)
    
    # Rotation matrices around x, y, and z axes
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))
    
    # Transform CARLA rotation to the target coordinate system:
    # In target: new_yaw = CARLA pitch, new_pitch = CARLA yaw, roll remains the same
    R_target = np.array([
        [0, 1, 0],  # Map CARLA x-axis to target y-axis
        [1, 0, 0],  # Map CARLA y-axis to target x-axis
        [0, 0, 1]   # Z-axis remains the same
    ])
    R_converted = np.dot(R_target, R)

    # Convert location
    x, y, z = location.x, location.y, location.z
    new_location = np.array([y, x, z])  # Swap x and y

    # Create the 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_converted
    transform_matrix[:3, 3] = new_location

    return transform_matrix

def compute_transform(sensor_transform_1, sensor_transform_2):
    """
    Compute the transformation matrix from sensor 1 coordinates to sensor 2 coordinates.
    
    Args:
        sensor_transform_1: CARLA Transform object for sensor 1.
        sensor_transform_2: CARLA Transform object for sensor 2.
    
    Returns:
        sensor1_to_sensor2: 4x4 transformation matrix from sensor 1 coordinates to sensor 2 coordinates.
    """
    # Get the world-to-sensor transformation matrices
    sensor1_to_world = get_transform_matrix(sensor_transform_1)
    sensor2_to_world = get_transform_matrix(sensor_transform_2)
    
    # Compute the inverse of the sensor2-to-world matrix to get world-to-sensor2
    world_to_sensor2 = np.linalg.inv(sensor2_to_world)
    
    # Compute the sensor1-to-sensor2 transformation by multiplying the matrices
    sensor1_to_sensor2 = np.dot(world_to_sensor2, sensor1_to_world)
    
    return sensor1_to_sensor2

def get_camera_intrinsics(sensor, width, height):
    """
    Calculate camera intrinsics matrix based on sensor attributes in CARLA.
    
    Args:
        sensor: CARLA camera sensor (provides FOV).
        width: Image width.
        height: Image height.
    
    Returns:
        4x4 camera intrinsics matrix.
    """
    # Get the horizontal field of view (FOV) from the sensor attributes
    fov = float(sensor.attributes['fov'])  # FOV in degrees
    
    # Convert FOV to radians
    fov_rad = np.deg2rad(fov)
    
    # Calculate focal length based on FOV and image dimensions
    focal_length_x = width / (2 * np.tan(fov_rad / 2))
    focal_length_y = height / (2 * np.tan(fov_rad / 2))  # Same focal length for square pixels
    
    # Assume principal point is at the center of the image
    c_x = width / 2
    c_y = height / 2

    # Camera intrinsics 3x3 matrix
    camera_intrinsics = np.array([
        [focal_length_x, 0, c_x],
        [0, focal_length_y, c_y],
        [0, 0, 1]
    ])

    # Convert to 4x4 matrix for homogeneous coordinates
    camera_intrinsics_4x4 = np.eye(4)
    camera_intrinsics_4x4[:3, :3] = camera_intrinsics

    return camera_intrinsics_4x4

