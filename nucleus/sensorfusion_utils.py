import open3d as o3d
import numpy as np

def read_pcd(pcd_filepath):
    '''
    Loads in pcd file and returns (N,3) or (N,4) numpy array

    Args:
        param pcd_filepath : filepath to local .pcd file
        type pcd_filepath: str

    Returns:
        point_numpy: (N,4) or (N,3) numpy array of points
        type point_numpy: np.array
    '''
    point_cloud = o3d.io.read_point_cloud(pcd_filepath)

    point_numpy = np.asarray(point_cloud.points)[:, :4]
    return point_numpy