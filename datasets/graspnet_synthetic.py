"""
TransCG Dataset.

Author: Hongjie Fang.
"""
import os
import json
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from utils.data_preparation import process_data
from tqdm import tqdm
import scipy.io as scio


class CameraInfo():
    """ Camera intrisics for point cloud creation. """

    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale


def transform_point_cloud(cloud, transform, format='4x4'):
    """ Transform points to new coordinates with transformation matrix.

        Input:
            cloud: [np.ndarray, (N,3), np.float32]
                points in original coordinates
            transform: [np.ndarray, (3,3)/(3,4)/(4,4), np.float32]
                transformation matrix, could be rotation only or rotation+translation
            format: [string, '3x3'/'3x4'/'4x4']
                the shape of transformation matrix
                '3x3' --> rotation matrix
                '3x4'/'4x4' --> rotation matrix + translation matrix

        Output:
            cloud_transformed: [np.ndarray, (N,3), np.float32]
                points in new coordinates
    """
    if not (format == '3x3' or format == '4x4' or format == '3x4'):
        raise ValueError('Unknown transformation format, only support \'3x3\' or \'4x4\' or \'3x4\'.')
    if format == '3x3':
        cloud_transformed = np.dot(transform, cloud.T).T
    elif format == '4x4' or format == '3x4':
        ones = np.ones(cloud.shape[0])[:, np.newaxis]
        cloud_ = np.concatenate([cloud, ones], axis=1)
        cloud_transformed = np.dot(transform, cloud_.T).T
        cloud_transformed = cloud_transformed[:, :3]
    return cloud_transformed


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def get_workspace_mask(cloud, seg, trans=None, organized=True, outlier=0):
    """ Keep points in workspace as input.

        Input:
            cloud: [np.ndarray, (H,W,3), np.float32]
                scene point cloud
            seg: [np.ndarray, (H,W,), np.uint8]
                segmantation label of scene points
            trans: [np.ndarray, (4,4), np.float32]
                transformation matrix for scene points, default: None.
            organized: [bool]
                whether to keep the cloud in image shape (H,W,3)
            outlier: [float]
                if the distance between a point and workspace is greater than outlier, the point will be removed

        Output:
            workspace_mask: [np.ndarray, (H,W)/(H*W,), np.bool]
                mask to indicate whether scene points are in workspace
    """
    if organized:
        h, w, _ = cloud.shape
        cloud = cloud.reshape([h * w, 3])
        seg = seg.reshape(h * w)
    if trans is not None:
        cloud = transform_point_cloud(cloud, trans)
    foreground = cloud[seg > 0]
    xmin, ymin, zmin = foreground.min(axis=0)
    xmax, ymax, zmax = foreground.max(axis=0)
    mask_x = ((cloud[:, 0] > xmin - outlier) & (cloud[:, 0] < xmax + outlier))
    mask_y = ((cloud[:, 1] > ymin - outlier) & (cloud[:, 1] < ymax + outlier))
    mask_z = ((cloud[:, 2] > zmin - outlier) & (cloud[:, 2] < zmax + outlier))
    workspace_mask = (mask_x & mask_y & mask_z)
    if organized:
        workspace_mask = workspace_mask.reshape([h, w])

    return workspace_mask


class GraspNetSyn(Dataset):
    """
    TransCG dataset.
    """
    def __init__(self, data_dir, camera = "realsense", split = 'train', **kwargs):
        """
        Initialization.

        Parameters
        ----------

        data_dir: str, required, the data path;

        split: str in ['train', 'test'], optional, default: 'train', the dataset split option.
        """
        super(GraspNetSyn, self).__init__()
        if split not in ['train', 'test']:
            raise AttributeError('Invalid split option.')
        self.data_dir = data_dir
        self.split = split

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.depthpath = []
        self.colorpath = []
        self.gt_depthpath = []
        self.labelpath = []
        self.metapath = []
        for x in tqdm(self.sceneIds, desc='Loading data path...'):
            for img_num in range(256):
                self.depthpath.append(os.path.join(data_dir, 'synthetic', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.gt_depthpath.append(os.path.join(data_dir, 'scenes', x, camera, 'depth', str(img_num).zfill(4) + '.png'))
                self.colorpath.append(os.path.join(data_dir, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.labelpath.append(os.path.join(data_dir, 'scenes', x, camera, 'label', str(img_num).zfill(4) + '.png'))
                self.metapath.append(os.path.join(data_dir, 'scenes', x, camera, 'meta', str(img_num).zfill(4) + '.mat'))

        # Other parameters
        self.use_aug = kwargs.get('use_augmentation', True)
        self.rgb_aug_prob = kwargs.get('rgb_augmentation_probability', 0.8)
        self.image_size = kwargs.get('image_size', (1280, 720))
        self.depth_min = kwargs.get('depth_min', 0.3)
        self.depth_max = kwargs.get('depth_max', 1.5)
        self.depth_norm = kwargs.get('depth_norm', 1.0)
        self.with_original = kwargs.get('with_original', False)

    def __getitem__(self, index):
        # img_path, camera_type, scene_type, perspective_id = self.sample_info[id]
        depth = np.array(Image.open(self.depthpath[index]), dtype = np.float32)
        height, width = depth.shape
        depth_gt = np.array(Image.open(self.gt_depthpath[index]), dtype = np.float32)
        seg = np.array(Image.open(self.labelpath[index]), dtype = np.uint8)
        meta = scio.loadmat(self.metapath[index])
        rgb = np.array(Image.open(self.colorpath[index]), dtype=np.float32)

        depth_coeff = meta['factor_depth']
        cam_intrinsic = meta['intrinsic_matrix'] # TODO resize
        camera = CameraInfo(width, height, cam_intrinsic[0][0], cam_intrinsic[1][1], cam_intrinsic[0][2], cam_intrinsic[1][2], depth_coeff)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        workspace_mask = get_workspace_mask(cloud, seg, organized=True, outlier=0.02)
        depth_gt_mask = np.asarray(workspace_mask, dtype=np.uint8)

        return process_data(rgb, depth, depth_gt, depth_gt_mask, cam_intrinsic, camera_type = 1,  split = self.split, image_size = self.image_size, depth_min = self.depth_min, depth_max = self.depth_max, depth_norm = self.depth_norm, use_aug = self.use_aug, rgb_aug_prob = self.rgb_aug_prob, inpainting = True, with_original = self.with_original)

    def __len__(self):
        return len(self.depthpath)
