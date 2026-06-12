import cv2
import numpy as np
import open3d as o3d
import argparse
from PIL import Image
from inference import Inferencer
import os
from tqdm import tqdm


CFG_BY_CAMERA = {
    "realsense": "/opt/data/private/TransCG/configs/inference_synthetic_realsense.yaml",
    "kinect": "/opt/data/private/TransCG/configs/inference_synthetic_kinect.yaml",
}

SCENE_RANGES = {
    "all": range(190),
    "test": range(100, 190),
}


def draw_point_cloud(color, depth, camera_intrinsics, use_mask = False, use_inpainting = True, scale = 1000.0, inpainting_radius = 5, fault_depth_limit = 0.2, epsilon = 0.01):
    """
    Given the depth image, return the point cloud in open3d format.
    The code is adapted from [graspnet.py] in the [graspnetAPI] repository.
    """
    d = depth.copy()
    c = color.copy() / 255.0

    if use_inpainting:
        fault_mask = (d < fault_depth_limit * scale)
        d[fault_mask] = 0
        inpainting_mask = (np.abs(d) < epsilon * scale).astype(np.uint8)
        d = cv2.inpaint(d, inpainting_mask, inpainting_radius, cv2.INPAINT_NS)

    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    xmap, ymap = np.arange(d.shape[1]), np.arange(d.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = d / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis = -1)

    if use_mask:
        mask = (points_z > 0)
        points = points[mask]
        c = c[mask]
    else:
        points = points.reshape((-1, 3))
        c = c.reshape((-1, 3))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(c.astype(np.float64))
    return cloud

def sample_inference():

    inferencer = Inferencer(cfg_path="/opt/data/private/TransCG/configs/inference_synthetic.yaml")

    rgb = np.array(Image.open('/opt/data/private/graspnet_dataset/scenes/scene_0021/realsense/rgb/0136.png'), dtype = np.float32)
    syn_depth = np.array(Image.open('/opt/data/private/graspnet_dataset/synthetic/scene_0021/realsense/depth/0136.png'), dtype = np.float32)
    # s2r_depth =
    depth_gt = np.array(Image.open('/opt/data/private/graspnet_dataset/scenes/scene_0021/realsense/depth/0136.png'), dtype = np.float32)

    syn_depth = syn_depth / 1000
    depth_gt = depth_gt / 1000

    s2r_depth, syn_depth = inferencer.inference(rgb, syn_depth, depth_coefficient = 3, inpainting = True)

    cam_intrinsics = np.load('/opt/data/private/graspnet_dataset/scenes/scene_0021/realsense/camK.npy')

    s2r_depth = np.clip(s2r_depth, 0.3, 1.0).astype(np.float32)
    syn_depth = np.clip(syn_depth, 0.3, 1.0)

    cloud = draw_point_cloud(rgb, s2r_depth, cam_intrinsics, scale = 1.0)
    cloud_gt = draw_point_cloud(rgb, depth_gt, cam_intrinsics, scale = 1.0)

    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    # sphere = o3d.geometry.TriangleMesh.create_sphere(0.002,20).translate([0,0,0.490])
    o3d.visualization.draw_geometries([cloud_gt])

def vis_s2r_depth():
    rgb = np.array(Image.open('/opt/data/private/graspnet_dataset/scenes/scene_0000/realsense/rgb/0010.png'), dtype = np.float32)
    s2r_depth = np.array(Image.open('/opt/data/private/graspnet_dataset/synthetic/scene_0000/realsense/s2r_depth_ddpm/0010.png'), dtype = np.float32)
    s2r_depth = s2r_depth / 1000
    cam_intrinsics = np.load('/opt/data/private/graspnet_dataset/scenes/scene_0021/realsense/camK.npy')
    cloud = draw_point_cloud(rgb, s2r_depth, cam_intrinsics, scale = 1.0)
    o3d.visualization.draw_geometries([cloud])

def infer_graspnet_syn(camera = "realsense", scene_split = "test"):
    camera = str.lower(camera)
    scene_split = str.lower(scene_split)
    if camera not in CFG_BY_CAMERA:
        raise ValueError("Unsupported camera: {}.".format(camera))
    if scene_split not in SCENE_RANGES:
        raise ValueError("Unsupported scene split: {}.".format(scene_split))
    inferencer = Inferencer(cfg_path=CFG_BY_CAMERA[camera])
    data_root = "/opt/data/private/graspnet_dataset"
    scene_list = ["scene_%04d"%i for i in SCENE_RANGES[scene_split]]
    for scene_name in scene_list:
        print(scene_name)
        rgb_dir = os.path.join(data_root, "scenes", scene_name, camera, "rgb")
        syn_depth_dir = os.path.join(data_root, "synthetic", scene_name, camera, "depth")
        s2r_depth_dir = os.path.join(data_root, "domain_translated", "transcg", scene_name, camera, "depth")
        if not os.path.exists(s2r_depth_dir):
            os.makedirs(s2r_depth_dir)
        for img_id in tqdm(range(256)):
            image_name = "%04d.png"%img_id
            rgb_path = os.path.join(rgb_dir, image_name)
            syn_depth_path = os.path.join(syn_depth_dir, image_name)
            s2r_depth_path = os.path.join(s2r_depth_dir, image_name)
            rgb = np.array(Image.open(rgb_path), dtype = np.float32)
            syn_depth = np.array(Image.open(syn_depth_path), dtype = np.float32) / 1000
            s2r_depth, syn_depth = inferencer.inference(rgb, syn_depth, depth_coefficient = 3, inpainting = True)
            s2r_depth = np.clip(s2r_depth, 0.3, 1.0).astype(np.float32)
            # s2r_depth 保存到 s2r_depth_path 文件类型是png 保证我下次调用如下
            # s2r_depth = np.array(Image.open(s2r_depth_path), dtype = np.float32) / 1000
            # 转换为毫米单位
            depth_mm = (s2r_depth * 1000).astype(np.uint16)

            # 保存为16bit PNG
            Image.fromarray(depth_mm).save(s2r_depth_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_split", choices = sorted(SCENE_RANGES.keys()), nargs = "?", default = "test")
    parser.add_argument("--camera", choices = sorted(CFG_BY_CAMERA.keys()), default = "realsense")
    args = parser.parse_args()
    infer_graspnet_syn(camera = args.camera, scene_split = args.scene_split)
