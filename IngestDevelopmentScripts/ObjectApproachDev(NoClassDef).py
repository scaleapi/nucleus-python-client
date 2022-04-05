from nucleus.transform import Transform
from nucleus.__init__ import NucleusClient
import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
import glob
import json

def read_pcd(fp):
   pcd = o3d.io.read_point_cloud(fp)
   points = np.asarray(pcd.points)
   return points

class RawPointCloud:
    def __init__(self, points: np.array = None, transform: Transform = None):
        self.points = points
        if transform is not None:
            self.points = transform.apply(points)

    def load_pcd(self, filepath: str, transform: Transform = None):
        points = read_pcd(filepath)
        self.points = points
        if transform is not None:
            self.points = transform.apply(points)

class CamCalibration:
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
        self.pose = Transform(matrix).inverse

    def __init__(self, extrinsic_matrix = None):
        self.extrinsic_matrix = extrinsic_matrix

class RawFrame:
    def __init__(self, dev_pose: Transform = None, **kwargs):
        self.dev_pose = pose
        self.items = {}
        for key, value in kwargs.items():
            self.items[key] = value

    def get_world_points(self):
        """Return the list of points with the frame transformation applied

        :returns: List of points in world coordinates
        :rtype: np.array

        """
        return np.hstack(
            [
                self.pose @ self.points[:, :3],
                self.points[:, 3:4],
                self.points[:, 4:5],
            ]
        )

class RawScene:
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
            RawScene.make_transforms_relative()
        for frame in self.frames:
            for item in frame.items:
                if isinstance(frame.items[item],RawPointCloud):
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


pcd_files = sorted(glob.glob("/Users/patrickbradley/Desktop/Pilots/Velodyne/Data/Velodyne 10 Frame Sequence/Pointclouds/*"))
poses = sorted(glob.glob("/Users/patrickbradley/Desktop/Pilots/Velodyne/Data/Velodyne 10 Frame Sequence/Poses/*"))

cam1_rotation = Quaternion(0.5020734498049307,0.5020734498049306,-0.4979179159268883,0.49791791592688817).rotation_matrix
cam1_translation = [-0.047619047619047894, -0.023809523809523725, -0.015873015873015817]

cam1_transform = Transform.from_Rt(R=cam1_rotation, t=cam1_translation)
cam_1_calib = CamCalibration(extrinsic_matrix=cam1_transform.matrix)

RawScene = RawScene()
for idx,pcd_fp in enumerate(pcd_files):
    pointcloud = RawPointCloud()
    pointcloud.load_pcd(pcd_fp)

    with open(poses[idx], "rb") as f:
        pose = json.load(f)
    pose = Transform.from_Rt(R=np.matrix(pose['rotation']), t=pose['translation'])

    raw_frame = RawFrame(lidar=pointcloud, cam1=cam_1_calib, dev_pose=pose)

    RawScene.add_frame(raw_frame)
RawScene.apply_transforms()











#

#
#





#
# RawPointCloud = RawPointCloud(
#     points = points,
#     transform = transform.
# )

#Local Image
# PrecursorImage1 = nucleus.PrecursorImage(
#     image_filepath="~/Desktop/ImageFile.png",
#     reference_id="scene-1-cam1-image0",
#     cam_params=cam1_cam_params
# )
#
# FramePose = nucleus.Transform.from_Rt(R=poses[idx][‘rotation’], t = poses[idx][‘translation’])
# PrecursorFrame = nucleus.PrecursorFrame(lidar=PrecursorPointCloud,
#                                         cam1 = PrecursorImage1,
#                                         cam2 = PrecursorImage2,
#                                         pose = FramePose)
#
# cam1_cam_transform = Transform.from_Rt(R=cam1_cam_extrinsics['rotation'], cam1_cam_extrinsics['translation'])
#
# #New Object - Intrinsics and Extrinsics (extrinsics are lidar to cam). Calibration is VERY similar to CameraParams but with added functionality
# cam1_cam_params = nucleus.Calibration(fx=Cam1_cam_intrinsics["fx"], fy=Cam1_cam_intrinsics['fy'], extrinsic_matrix=cam1_cam_transform.matrix)
#
# PrecursorScene = LidarSceneTransformer()
#
# local_pcds = sorted(glob.glob("~/localpcdfiles"))
# for idx, pcd_fp in enumerate(local_pcds):
#     points = read_pcd(fp) #Points in Ego Coordinates
#
#     PrecursorPointCloud = nucleus.PrecursorPointCloud(
#         points = points,
#         reference_id="scene-1-pointcloud-0",
#         metadata={"is_raining": False}
#     )
#
#     #Local Image
#     PrecursorImage1 = nucleus.PrecursorImage(
#         image_filepath="~/Desktop/ImageFile.png",
#         reference_id="scene-1-cam1-image0",
#         cam_params=cam1_cam_params
#     )
#
#     FramePose = nucleus.Transform.from_Rt(R=poses[idx][‘rotation’], t = poses[idx][‘translation’])
#     PrecursorFrame = nucleus.PrecursorFrame(lidar=PrecursorPointCloud,
#                                             cam1 = PrecursorImage1,
#                                             cam2 = PrecursorImage2,
#                                             pose = FramePose)
#
#     PrecursorScene.add_frame(PrecursorFrame)
#
# final_scene = ingest_scene.transform_and_upload(s3_client = s3_client,
#                                                 bucket = "bucket",
#                                                 prefix = "prefix",
#                                                 reference_id = "scene-0")
#
# job = dataset.append(
#     items=[final_scene],
#     update=True,
#     asynchronous=True  # required for 3D uploads
# )