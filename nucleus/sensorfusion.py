import numpy as np
from nucleus.transform import Transform
from nucleus.utils import read_pcd
import copy

class RawPointCloud:
    '''
    RawPointClouds are containers for raw point cloud data. This structure contains Point Clouds as (N,3) or (N,4) numpy arrays
    Point cloud data is assumed to be in ego coordinates.
    If the point cloud is in World Coordinates, one can provide the inverse pose as the "transform" argument. If this argument
    is present, the point cloud will be transformed back into ego coordinates
    Args:
        points (np.array): Point cloud data represented as (N,3) or (N,4) numpy arrays
        transform (:class:`Transform`): If in World Coordinate, the transformation used to transform lidar points into ego coordinates
    '''

    def __init__(self, points: np.array = None, transform: Transform = None):
        self.points = points
        if transform is not None:
            self.points = transform.apply(points)

    def load_pcd(self, filepath: str, transform: Transform = None):
        '''
        Takes raw pcd file and reads in as numpy array.
        Args:
            filepath (str): Local filepath to pcd file
            transform (:class:`Transform`): If in world, the transformation used to transform lidar points into ego coordinates
        '''
        points = read_pcd(filepath)
        self.points = points
        if transform is not None:
            self.points = transform.apply(points)


class CameraCalibration:
    '''
    CamCalibration solely holds the pose of the camera
    This CamCalibration will inevitably be transformed by the device_pose
    Args:
            extrinsic_matrix (np.array): (4,4) extrinsic transformation matrix representing device_to_lidar
    '''

    def __init__(self, extrinsic_matrix=None):
        self.extrinsic_matrix = extrinsic_matrix

    @property
    def extrinsic_matrix(self):
        """Camera extrinsic
        :getter: Return camera's extrinsic matrix (pose.inverse[:3, :4])
        :setter: pose = Transform(matrix).inverse
        :type: 3x4 matrix
        """
        return self.pose.inverse[:3, :4]

    @extrinsic_matrix.setter
    def extrinsic_matrix(self, matrix):
        '''
        Sets pose as inverse of extrinsic matrix
        '''
        self.pose = Transform(matrix).inverse


class RawFrame:
    '''
    RawFrames are containers for point clouds, image extrinsics, and device pose.
    These objects most notably are leveraged to transform point clouds and image extrinsic matrices to the world coordinate frame.
    Args:
            dev_pose (:class:`Transform`): World Coordinate transformation for frame
            **kwargs (Dict[str, :class:`RawPointCloud, :class: CameraCalibration`]): Mappings from sensor name
              to pointcloud and camera calibrations. Each frame of a lidar scene must contain exactly one
              pointcloud and any number of camera calibrations
    '''
    def __init__(self, dev_pose: Transform = None, **kwargs):
        self.dev_pose = dev_pose
        self.items = {}
        for key, value in kwargs.items():
            if isinstance(value, CamCalibration):
                self.items[key] = copy.copy(value)
            else:
                self.items[key] = value

    def get_world_points(self):
        """Return the list of points with the frame transformation applied
        :returns: List of points in world coordinates
        :rtype: np.array
        """
        for item in self.items:
            if isinstance(self.items[item], RawPointCloud):
                return np.hstack(
                    [
                        self.dev_pose @ self.items[item][:, :3],
                        self.items[item][:, 3:4],
                        self.items[item][:, 4:5]
                    ]
                )


class RawScene:
    '''
    RawsScenes are containers for frames
    These objects most notably are leveraged to transform point clouds and image extrinsic matrices to the world coordinate frame.
    Args:
            dev_pose (:class:`Transform`): World Coordinate transformation for frame
            **kwargs (Dict[str, :class:`RawPointCloud, :class: CameraCalibration`]): Mappings from sensor name
              to pointcloud and camera calibrations. Each frame of a lidar scene must contain exactly one
              pointcloud and any number of camera calibrations
    '''
    def __init__(self, frames: [RawFrame] = None):
        if frames is None:
            self.frames = []
        else:
            self.frames = frames

    def add_frame(self, frame: RawFrame = None):
        self.frames.append(frame)

    def make_transforms_relative(self):
        """Make all the frame transform relative to the first transform/frame. This will set the first transform to position (0,0,0) and heading (1,0,0,0)"""
        offset = self.frames[0].dev_pose.inverse
        for frame in self.frames:
            frame.dev_pose = offset @ frame.dev_pose

    def apply_transforms(self, relative: bool = False):
        if relative:
            self.make_transforms_relative()
        for frame in self.frames:
            for item in frame.items:
                if isinstance(frame.items[item], RawPointCloud):
                    frame.items[item].points = np.hstack(
                        [
                            frame.dev_pose @ frame.items[item].points[:, :3],
                            frame.items[item].points[:, 3:4],
                            frame.items[item].points[:, 4:5],
                        ]
                    )
                else:
                    wct = frame.dev_pose @ frame.items[item].pose
                    frame.items[item].pose = wct