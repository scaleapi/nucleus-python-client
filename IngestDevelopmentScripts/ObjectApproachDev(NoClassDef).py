from nucleus.sensorfusion import *
import numpy as np
import glob

def format_pointcloud(lidar_np):
    mask = lidar_np[:, 4] == 0.0
    pc_1 = lidar_np[mask, :]
    pc_1 = np.delete(pc_1, (4), 1)
    return pc_1

npz_files = sorted(glob.glob("sdk-sample-data/*"))
npz_files.remove("sdk-sample-data/extrinsics_seq000-003_11042021.npz")

updated_extrinsics = np.load("sdk-sample-data/extrinsics_seq000-003_11042021.npz", allow_pickle = True)
wnsl_extrinsics = updated_extrinsics['camera_WindshieldNarrowStereoLeft_lidar_extrinsic']
print(f"Camera Extrinsics: \n{wnsl_extrinsics}")

cam_1_calib = CamCalibration(extrinsic_matrix=wnsl_extrinsics)
print(f"Original Camera Pose:{cam_1_calib.pose}")

RawScene = RawScene()
for idx,npz_fp in enumerate(npz_files):
    print(f"Frame Index: {idx}")

    frame_npz = np.load(npz_fp, allow_pickle=True)

    pointcloud_np= format_pointcloud(frame_npz['points'])
    print(f"PointCloud Shape: {pointcloud_np.shape}")

    pointcloud = RawPointCloud(points=pointcloud_np)

    print(f"World Coordinate Transformation:\n{frame_npz['vehicle_local_tf']}")
    frame_pose = Transform(frame_npz['vehicle_local_tf'])
    print(f"Frame Pose: {frame_pose}")

    raw_frame = RawFrame(lidar=pointcloud, cam1=cam_1_calib, dev_pose=frame_pose)
    RawScene.add_frame(raw_frame)

print(f"Frame 5, Point1 PreTransform: {RawScene.frames[4].items['lidar'].points[0]}")
print(f"Frame 5, Camera Extrinsics PreTransform: {RawScene.frames[4].items['cam1'].pose}")
RawScene.apply_transforms(relative=True)
print(f"Frame 5, Point1 in World: {RawScene.frames[4].items['lidar'].points[0]}")
print(f"Frame 5, Camera Extrinsics PostTransform: {RawScene.frames[4].items['cam1'].pose}")






