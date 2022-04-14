import numpy as np
import os


def get_dataset_root(root_list, file_name=None):
    print(root_list)
    roots = root_list.split(";")
    root = None
    for rt in roots:
        if (file_name is None and os.path.exists(rt)) or (file_name is not None and os.path.exists(os.path.join(rt, file_name))):
            root = rt
            break
    if root is None:
        raise ValueError(f"Incorrect dataset root paths: {roots}")
    return root

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def plane_create_primitive_from_dict(d):
    assert d['type'] == 'plane'
    location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
    axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
    return {"type": "plane", "plane_n": axis, "plane_c": np.dot(location, axis)}

def sphere_create_primitive_from_dict(d):
    assert d['type'] == 'sphere'
    location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
    radius = float(d['radius'])
    return {"type": "sphere", "sphere_center": location, "sphere_radius": radius}

def cylinder_create_primitive_from_dict(d):
    assert d['type'] == 'cylinder'
    location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
    axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
    radius = float(d['radius'])
    return {"type": "cylinder", "cylinder_center": location, "cylinder_radius": radius, "cylinder_axis": axis}

def cone_create_primitive_from_dict(d):
    assert d['type'] == 'cone'
    apex = np.array([d['apex_x'], d['apex_y'], d['apex_z']], dtype=float)
    axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
    half_angle = float(d['semi_angle'])
    return {"type": "cone", "cone_apex": apex, "cone_axis": axis, "cone_half_angle": half_angle}


NAME_TO_FITTER_DICT = {
    'plane': plane_create_primitive_from_dict, # 0
    'sphere': sphere_create_primitive_from_dict, # 1
    'cylinder': cylinder_create_primitive_from_dict, # 2
    'cone': cone_create_primitive_from_dict, # 3
}

NAME_TO_ID_DICT = {
    'plane': 0, # 0
    'sphere': 1, # 1
    'cylinder': 2, # 2
    'cone': 3, # 3
}

NAME_TO_REMAIN_PARAM = {
    'plane': ['plane_n'],
    'sphere': ['sphere_center'],
    'cylinder': ['cylinder_axis'],
    'cone': ['cone_axis']
}


def create_primitive_from_dict(meta_dict):
    return NAME_TO_FITTER_DICT[meta_dict['type']](meta_dict)

def extract_parameter_data_as_dict_plane(primitives, n_max_instances):
    n = np.zeros(dtype=float, shape=[n_max_instances, 3])
    for i, primitive in enumerate(primitives):
        if primitive['type'] == 'plane':

            n[i] = primitive['plane_n']
    return {
        'plane_n': n
    }

def extract_parameter_data_as_dict_sphere(primitives, n_max_instances):
    n = np.zeros(dtype=float, shape=[n_max_instances, 3])
    return {
        'sphere_center': n
    }

def extract_parameter_data_as_dict_cylinder(primitives, n_max_primitives):
    n = np.zeros(dtype=float, shape=[n_max_primitives, 3])
    for i, primitive in enumerate(primitives):
        if primitive['type'] == 'cylinder':
            n[i] = primitive['cylinder_axis']
    return {
        'cylinder_axis': n
    }

def extract_parameter_data_as_dict_cone(primitives, n_max_instances):
    axis_gt = np.zeros(dtype=float, shape=[n_max_instances, 3])
    for i, primitive in enumerate(primitives):
        if primitive['type'] == 'cone':
            axis_gt[i] = primitive['cone_axis']
    return {
        'cone_axis': axis_gt,
    }


NAME_TO_PARAM_EXTRACT_DICT = {
    'plane': extract_parameter_data_as_dict_plane, # 0
    'sphere': extract_parameter_data_as_dict_sphere, # 1
    'cylinder': extract_parameter_data_as_dict_cylinder, # 2
    'cone': extract_parameter_data_as_dict_cone, # 3
}


def extract_parameter_data_as_dict(instances, n_max_instances, type):
    return NAME_TO_PARAM_EXTRACT_DICT[type](instances, n_max_instances)


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
      rotation is per shape based along up direction
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data.astype(np.float32)


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data.astype(np.float32)


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.30):
    """ Randomly perturb the point clouds by small rotations
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data.astype(np.float32)


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data.astype(np.float32)


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data.astype(np.float32)


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data

class Augment:
    def __init__(self, ):
        pass

    def augment(self, batch_data):
        if np.random.random() > 0.7:
            batch_data = rotate_perturbation_point_cloud(batch_data)
        if np.random.random() > 0.7:
            batch_data = jitter_point_cloud(batch_data)
        if np.random.random() > 0.7:
            batch_data = shift_point_cloud(batch_data, 0.05)
        if np.random.random() > 0.7:
            batch_data = random_scale_point_cloud(batch_data)
        return batch_data
