import torch
import torch.nn as nn
import numpy as np
from mmdet.models import HEADS
from mmcv.runner import BaseModule
from .modules.utils import Voxelization
import spconv.pytorch as spconv
import pickle
import os
import cv2

@HEADS.register_module()
class VoxelProposalLayer(BaseModule):
    def __init__(
        self,
        point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
        input_dimensions=[256, 256, 32],
        data_config=None,
        init_cfg=None,
        factor=4,
        **kwargs
    ):
        super(VoxelProposalLayer, self).__init__(init_cfg)

        self.data_config = data_config
        self.init_cfg = init_cfg
        self.voxelize = Voxelization(
            point_cloud_range=point_cloud_range, 
            spatial_shape=np.array(input_dimensions))
        
        image_grid = self.create_grid()
        self.register_buffer('image_grid', image_grid)

        self.input_dimensions = input_dimensions
        self.factor = factor
        
    def create_grid(self):
        """
        descript
            make grid in image plane
        params
            None
        returns
            - grid (tensor) | (B, N, H, W)
        """

        ogfH, ogfW = self.data_config['input_size']
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)

        grid = torch.stack((xs, ys), 1)
        return nn.Parameter(grid, requires_grad=False)
    
    def depth2lidar(self, image_grid, stereo_depth, img_inputs, cur_cam2world):
        """
        descript
            lift image pixel to lidar coordinate by using depth estimation depth map
        params
            - image (tensor): image                             | (B, N, 3, H, W)
            - image_grid (tensor): image grid index             | (B, 2, H, W)
            - depth (tensor): depth value                       | (B, 1, H, W)
            - cam_params (list): list of cam parmas             | info: {'rots', 'trans', 'intrins', 'post_rots', 'post_trans', 'bda'}
            - is_hist (bool): if True -> calc history lidar pcd | 
            - cur_pose (tensor)
        returns
            - points (tensor): point cloud on lidar coordinates | (B, N, 3)
            - points_rgb (tensor): rgb of each point cloud      | (B, N, 3)
        """
        num_frames = len(stereo_depth)
        cur_idx = num_frames-1
        
        b, _, h, w = stereo_depth[-1].shape
        _, rots, trans, intrins, post_rots, post_trans, bda = img_inputs[0][:7]
        
        lidar_points_list = []
        for i, (depth, img_input) in enumerate(zip(stereo_depth, img_inputs)):
            hist_cam2world = img_input[-1]
            
            cur_R = cur_cam2world[..., :3, :3]
            cur_T = cur_cam2world[..., :3, 3:]
            hist_R = hist_cam2world[..., :3, :3]
            hist_T = hist_cam2world[..., :3, 3:]
            
            if i == cur_idx:
                image_grid_inter_x = torch.nn.functional.interpolate(image_grid[:, 0:1], size=(image_grid.shape[2]*self.factor, image_grid.shape[3]*self.factor),
                                                                     mode='bilinear', align_corners=False)
                image_grid_inter_y = torch.nn.functional.interpolate(image_grid[:, 1:2], size=(image_grid.shape[2]*self.factor, image_grid.shape[3]*self.factor),
                                                                     mode='bilinear', align_corners=False)
                image_grid_inter_z = torch.nn.functional.interpolate(depth, size=(depth.shape[2]*self.factor, depth.shape[3]*self.factor),
                                                                     mode='bilinear', align_corners=False)
                image_grid_inter = torch.cat([image_grid_inter_x, image_grid_inter_y], dim=1)
                
                points = torch.cat([image_grid_inter.repeat(b, 1, 1, 1), image_grid_inter_z], dim=1)
                points = points.view(b, 3, h * w * self.factor * self.factor).permute(0, 2, 1)
                
            else:
                points = torch.cat([image_grid.repeat(b, 1, 1, 1), depth], dim=1)
                points = points.view(b, 3, h * w).permute(0, 2, 1)
                
            # undo pos-transformer
            points = points - post_trans.view(b, 1, 3)
            points = torch.inverse(post_rots).view(b, 1, 3, 3).matmul(points.unsqueeze(-1))
            
            points = torch.cat([points[:, :, 0:2, :] * points[:, :, 2:3, :], points[:, :, 2:3, :]], dim=2)
            
            shift = intrins[..., :3, 3]
            points = points - shift.view(b, 1, 3, 1)
            intrins_rot = intrins[..., :3, :3]
            points = torch.inverse(intrins_rot) @ points
            
            points = hist_R @ points + hist_T
            points = torch.inverse(cur_R) @ (points - cur_T)
            points = (rots @ points).squeeze(-1) + trans
            
            if bda.shape[-1] == 4:
                points = torch.cat((points, torch.ones(*points.shape[:-1], 1).type_as(points)), dim=-1)
                points = bda.view(b, 1, 4, 4).matmul(points.unsqueeze(-1)).squeeze(-1)
                points = points[..., :3]
            else:
                points = bda.view(b, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        
            lidar_points_list.append(points)
            
        total_lidar_pcd = torch.cat(lidar_points_list, dim=1)
            
        
        return total_lidar_pcd, lidar_points_list
        
    
    def lidar2voxel(self, points, device):
        """
        descript
            lidar pcd to voxel (Voxelization)
        params
            - points (tensor): point cloud on lidar coordinate  | [B, N, 3]
            - device (device): gpu/cpu
        returns
            - unq (tensor): unique voxel index which occupied   | [N, 4]
            - unq_inv (tensor): ?
        """
        points_reshape = []
        batch_idx = []
        tensor = torch.ones((1,), dtype=torch.long).to(device)

        for i, pc in enumerate(points):
            points_reshape.append(pc)
            batch_idx.append(tensor.new_full((pc.shape[0],), i))
        
        points_reshape, batch_idx = torch.cat(points_reshape), torch.cat(batch_idx) # points_reshape: [N, 3] - all points / batch_idx: [N] - batch idx for each point

        unq, unq_inv = self.voxelize(points_reshape, batch_idx)

        return unq, unq_inv
    
    def forward(self, stereo_depth, img_inputs):
        """
        params
            - cam_params(list): current img inputs / camera params  | info: {'img', 'rot', 'trans', 'intrin', 'post_rot', 'post_trans', 'bda_rot', 'cam2lidar', 'cam2world'}
            - img_metas(dict): current img's meta data              | keys: {'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape', 'gt_depths'}
            - post_cam_params(list): post imgs inputs / cam paras   | info: {'img', 'rot', 'trans', 'intrin', 'post_rot', 'post_trans', 'bda_rot', 'cam2lidar', 'cam2world'}
            - post_img_metas(dict): post imgs meta data             | keys: {'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape', 'gt_depths'}
            
        returns
            - input (Tensor): sparse voxel grid <occupied 1 / non 0>            | [B, N, H, W, Z]
            - lidar_pcd_coord (Tensor): total point clouds on lidar coordinates | [B, N, 3]
            - lidar_pcd_rgb (Tensor): rgb value for each point clouds           | [B, N, 3]
        """
        
        cur_cam2world = img_inputs[-1][-1]
        total_lidar_pcd, lidar_pcd_list = self.depth2lidar(self.image_grid, stereo_depth, img_inputs, cur_cam2world)
        
        unq, unq_inv = self.lidar2voxel(total_lidar_pcd, total_lidar_pcd.device)
        
        sparse_tensor = spconv.SparseConvTensor(
            torch.ones(unq.shape[0], dtype=torch.float32).view(-1, 1).to(total_lidar_pcd.device),
            unq.int(), spatial_shape=self.input_dimensions, batch_size=(torch.max(unq[:, 0] + 1))
        )
        
        proposal = sparse_tensor.dense()

        return proposal, lidar_pcd_list