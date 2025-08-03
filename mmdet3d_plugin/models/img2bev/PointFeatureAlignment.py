import torch
import torch.nn as nn
import numpy as np
from mmdet.models import HEADS
from mmcv.runner import BaseModule
from .modules.utils import Voxelization
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

@HEADS.register_module()
class PointFeatureAlignment(BaseModule):
    def __init__(self,
                 n_points=4,
                 img_size=(384, 1280),
                 pcd_range=[0, -25.6, -2, 51.2, 25.6, 4.4],
                 spatial_shape=(128, 128, 16),
                 dim=128,
                 factor=4,
                 encoder=None):
        
        super(PointFeatureAlignment, self).__init__()
        
        self.n_points = n_points
        self.img_size = img_size
        self.pcd_range = pcd_range
        self.spatial_shape = spatial_shape
        self.coords_range_xyz = np.array([
            [self.pcd_range[0], self.pcd_range[3]],
            [self.pcd_range[1], self.pcd_range[4]],
            [self.pcd_range[2], self.pcd_range[5]]
        ])
        self.dim = dim
        self.factor = factor
        
    def mask_op(self, data, x_min, x_max):
        mask = (data > x_min) & (data < x_max)
        return mask
        
    def masking(self, data):
        coords_range_xyz = np.array([
            [self.pcd_range[0], self.pcd_range[3]],
            [self.pcd_range[1], self.pcd_range[4]],
            [self.pcd_range[2], self.pcd_range[5]]
        ])
        
        eps = 0.0001
        
        mask_x = self.mask_op(data[..., 0], coords_range_xyz[0][0] + eps, coords_range_xyz[0][1] - eps)
        mask_y = self.mask_op(data[..., 1], coords_range_xyz[1][0] + eps, coords_range_xyz[1][1] - eps)
        mask_z = self.mask_op(data[..., 2], coords_range_xyz[2][0] + eps, coords_range_xyz[2][1] - eps)
        
        return mask_x & mask_y & mask_z
    
    def sparse_quantize(self, pcd, coords_range, spatial_shape):
        idx = spatial_shape * (pcd - coords_range[0]) / (coords_range[1] - coords_range[0])
        return idx.long()

    def linear_inverse(self, x, x_min=0.0, x_max=70.0):
        norm_x = (x - x_min) / (x_max - x_min)  # Normalize to [0, 1]
        inv_x = 1.0 - norm_x  # Inverse so large x → small value
        return torch.clamp(inv_x, 0.0, 1.0)

    def forward(self, img_feats, stereo_depth, lidar_pcd_list):
        """_summary_

        Args:
            img_feats (_type_): _description_
            lidar_pcd (_type_): _description_
        """

        B, N, C, H, W = img_feats.shape
        
        chunks = torch.chunk(img_feats, chunks=N, dim=1)
        upsampled_chunks = [
            torch.nn.functional.interpolate(chunk.squeeze(0), size=self.img_size, mode='bilinear', align_corners=False)
            for chunk in chunks
        ]
        
        point_feature_list = []
        for i, (img_feat, depth, lidar_pcd) in enumerate(zip(upsampled_chunks, stereo_depth, lidar_pcd_list)):
            if i == N-1:
                img_feat = torch.nn.functional.interpolate(img_feat, size=(self.img_size[0]*self.factor, self.img_size[1]*self.factor), mode='bilinear',
                                                           align_corners=False)
                
            mask = self.masking(lidar_pcd)
            img_feat_flatten = img_feat.unsqueeze(0).flatten(3).permute(0, 1, 3, 2)
            
            if i != N-1:
                depth_flatten = depth.flatten(2)
                depth_weight = self.linear_inverse(depth_flatten,
                                                   depth_flatten.min(),
                                                   depth_flatten.max()).unsqueeze(-1)
                img_feat_flatten = img_feat_flatten * depth_weight
            
            filter_lidar_pcd = lidar_pcd[mask]
            filter_feats_flatten = img_feat_flatten.squeeze(0)[mask]
            
            xidx = self.sparse_quantize(filter_lidar_pcd[:, 0], self.coords_range_xyz[0], self.spatial_shape[0])
            yidx = self.sparse_quantize(filter_lidar_pcd[:, 1], self.coords_range_xyz[1], self.spatial_shape[1])
            zidx = self.sparse_quantize(filter_lidar_pcd[:, 2], self.coords_range_xyz[2], self.spatial_shape[2])
        
            xyz_idx = torch.stack([xidx, yidx, zidx], dim=-1).long()
            
            voxel_volume = torch.zeros((*self.spatial_shape, self.dim),
                                       dtype=filter_feats_flatten.dtype, device=filter_feats_flatten.device)
            voxel_volume_flatten = voxel_volume.view(-1, self.dim)
            xyz_idx_flatten = xyz_idx[:, 0] * self.spatial_shape[1] * self.spatial_shape[2] + xyz_idx[:, 1] * self.spatial_shape[2] + xyz_idx[:, 2]
            voxel_volume_flatten.index_add_(0, xyz_idx_flatten, filter_feats_flatten)
            voxel_volume = voxel_volume_flatten.view(*self.spatial_shape, -1)
            
            point_feature_list.append(voxel_volume.unsqueeze(0))
            
        output_volume = torch.cat(point_feature_list, dim=0)
        output_volume = output_volume.mean(dim=0).unsqueeze(0).permute(0, 4, 1, 2, 3)
        
        return output_volume