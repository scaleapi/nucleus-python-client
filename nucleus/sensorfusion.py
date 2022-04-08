import numpy as np
from nucleus.transform import Transform
import copy
from typing import List
from dataclasses import dataclass

from sensorfusion_utils import read_pcd

@dataclass
class RawPointCloud:
    '''
    RawPointClouds are containers for raw point cloud data. This structure contains Point Clouds as (N,3) or (N,4) numpy arrays
    Point cloud data is assumed to be in ego coordinates.

    If the point cloud is in world coordinates, one can provide the inverse pose as the "transform" argument. If this argument
    is present, the point cloud will be transformed back into ego coordinates by:

        transform.apply(points)

    or in the extended implementation:

        points_4d = np.hstack([points[:, :3], np.ones((points.shape[0], 1))])
        transformed_4d = points_4d.dot(self.matrix.T)
        return np.hstack([transformed_4d[:, :3], points[:, 3:]])


    Args:
        points (np.array): Point cloud data represented as (N,3) or (N,4) numpy arrays
        transform (:class:`Transform`): If in World Coordinate, the transformation used to transform lidar points into ego coordinates
    '''
    points: np.array

    def __init__(self, points: np.array = None, transform: Transform = None):
        self.points = points

        if points is not None and (len(self.points.shape) != 2 or self.points.shape[1] not in [3, 4]):
            raise Exception(f'numpy array has unexpected shape{self.points.shape}. Please convert to (N,3) or (N,4) numpy array where each row is [x,y,z] or [x,y,z,i]')

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
    CameraCalibration solely holds the pose of the camera
    This CameraCalibration will inevitably be transformed by the device_pose
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
            device_pose (:class:`Transform`): World Coordinate transformation for frame
            **kwargs (Dict[str, :class:`RawPointCloud, :class: CameraCalibration`]): Mappings from sensor name
              to pointcloud and camera calibrations. Each frame of a lidar scene must contain exactly one
              pointcloud and any number of camera calibrations
    '''
    def __init__(self, device_pose: Transform = None, **kwargs):
        self.device_pose = device_pose
        self.items = {}
        for key, value in kwargs.items():
            if isinstance(value, CameraCalibration):
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
                        self.device_pose @ self.items[item][:, :3],
                        self.items[item][:, 3:4],
                        self.items[item][:, 4:5]
                    ]
                )


class RawScene:
    '''
    RawsScenes are containers for frames
    Args:
            frames (:class:`RawFrame`): Indexed sequential frame objects composing the scene
    '''

    def __init__(self, frames: List[RawFrame] = []):
        if frames is None:
            self.frames = []
        else:
            self.frames = frames

    def add_frame(self, frame: RawFrame):
        self.frames.append(frame)

    def make_transforms_relative(self):
        """Make all the frame transform relative to the first transform/frame. This will set the first transform to position (0,0,0) and heading (1,0,0,0)"""
        offset = self.frames[0].device_pose.inverse
        for frame in self.frames:
            frame.device_pose = offset @ frame.device_pose

    def apply_transforms(self, relative: bool = False):
        if relative:
            self.make_transforms_relative()
        for frame in self.frames:
            for item in frame.items:
                if isinstance(frame.items[item], RawPointCloud):
                    frame.items[item].points = np.hstack(
                        [
                            frame.device_pose @ frame.items[item].points[:, :3],
                            frame.items[item].points[:, 3:4],
                            frame.items[item].points[:, 4:5],
                        ]
                    )
                else:
                    wct = frame.device_pose @ frame.items[item].pose
                    frame.items[item].pose = wct