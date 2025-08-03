import torch
import torch.nn as nn
import numpy as np
from mmdet.models import HEADS
from mmcv.runner import BaseModule

@HEADS.register_module()
class ExtractTurningViewLayer(BaseModule):
    def __init__(self,
                 angle=30,
                 trans_side=0.,
                 resolution=None,
                 pcd_range=None,
                 init_cfg=None):
        
        super(ExtractTurningViewLayer, self).__init__(init_cfg)
        
        self.angle = angle
        self.trans_side = trans_side
        self.resolution = resolution
        self.pcd_range = pcd_range
        
    def proj_pcd2plane(self, lidar_pcd, K, rots, trans, post_rots, post_trans, theta=None):
        """
        Descript:
            Project lidar point cloud to image plane
        Args:
            - lidar_pcd (Tensor): Total lidar point clouds                  | [B, N, 3]
            - K (Tensor) : Intrinsic matrix                                 | [B, N, 4, 4]
            - post_rots (Tensor): Image augmentation rotation matrix        | [B, N, 3, 3]
            - post_trans (Tensor): Image augmentation translation matrix    | [B, N, 3]
            - proj_view (string): projection view type (left, right)
            - theta (scalar): turning view angle
        Returns:
            - turning_img (Tensor): turning view pcd on image plane         | [B, N, 3]
            - mask (Tensor): masking which out of boundary                  | [B, N]
            - depth (Tensor): depth value of projection pcd on plane        | [B, N]
        """
        H, W = self.resolution
        
        turning_extrinsic = torch.Tensor([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])  # turning_extrinsic: [3, 3]
        turning_extrinsic = turning_extrinsic.unsqueeze(0).unsqueeze(0).to(lidar_pcd.device)    # turning_extrinsic: [B, N, 3, 3]
        
        # TODO: + trans_side 가 맞는지 check 필요
        cam_pcd = (torch.inverse(rots) @ lidar_pcd.unsqueeze(-1) - trans.unsqueeze(-1)).squeeze(-1) # cam_pcd: [B, N, 3]
        turning_cam_pcd = ((turning_extrinsic @ cam_pcd.unsqueeze(-1)) + torch.Tensor([0, 0, self.trans_side]).to(lidar_pcd.device).reshape(1, 1, -1, 1)).squeeze(-1)   # turning_image: [B, N, 3]
        turning_img = (K[..., :3, :3] @ turning_cam_pcd.unsqueeze(-1) + K[..., :3, 3:]).squeeze(-1) # turning_img: [B, N, 3]
        
        turning_img[..., 0] /= turning_img[..., 2]
        turning_img[..., 1] /= turning_img[..., 2]
        depth = turning_img[..., 2]
        
        turning_img[..., :2] = (post_rots[..., :2, :2] @ turning_img[..., :2].unsqueeze(-1)).squeeze(-1) + post_trans[..., :2]
        
        # TODO: depth > 1 인지 depth > 0 인지 check 필요
        mask = (turning_img[..., 0] > 0) & (turning_img[..., 0] < W) & (turning_img[..., 1] > 0) & (turning_img[..., 1] < H) & (depth > 0)        
        
        return turning_img, mask, depth
    
    def sample_image(self, turning_pcd, depth, pcd_rgb, mask):
        """
        Descript:
            Sample image
        Args:
            turning_img_pcd (Tensor): point cloud on turing image plane | [B, N, 3]
            depth (Tensor): depth value                                 | [B, N]
            pcd_rgb (Tensor): rgb value of depth                        | [B, N, 3]
            mask (Tensor): masking where points are out of boundary     | [B, N]
        Returns:
            image (Tensor): Sampled image                               | [B, N, 3, H, W]
        """
        H, W = self.resolution
        
        # Discard points which are out of boundary
        valid_turning_pcd = turning_pcd[mask]   # valid_turning_pcd: [N, 3]
        valid_depth = depth[mask]               # valid_depth: [N]
        valid_rgb = pcd_rgb[mask]               # valid_rgb: [N, 3]
        
        valid_turning_pcd = valid_turning_pcd.floor().type(torch.long)  # valid_turning_pcd: [N, 3]
        
        # Sorting depth value
        _, sorted_idx = torch.sort(valid_depth, descending=True)    # sorted_idx: [N]
        sorted_pcd = valid_turning_pcd[sorted_idx]                  # sorted_pcd: [N, 3]
        sorted_rgb = valid_rgb[sorted_idx]                          # sorted_rgb: [N, 3]
        
        # Sampling image
        image = torch.zeros((H, W, 3)).to(pcd_rgb.device)           # image: [H, W, 3]
        image[sorted_pcd[:, 1], sorted_pcd[:, 0]] = sorted_rgb      # image: [H, W, 3]
        image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)    # image: [B, N, 3, H, W]
        
        return image
        
    def forward(self, front_image, lidar_pcd, lidar_pcd_rgb, intrinsic, rots, trans, post_rots, post_trans):
        """
        Descript:
            Extract turning view layer which controlled by angle
        Args:
            - front_image (Tensor): current front image                     | [B, N, 3, H, W]
            - lidar_pcd (Tensor): Total lidar point cloud                   | [B, N, 3]
            - lidar_pcd_rgb (Tensor): rgb value of each total point cloud   | [B, N, 3]
            - intrinsic (Tensor): camera intrinsic matrix                   | [B, N, 4, 4]
            - rots (Tensor): cam2lidar rotation matrix                      | [B, N, 3, 3]
            - trans (Tensor): cam2lidar translation matrix                  | [B, N, 3]
            - post_rots (Tensor): augmentation rotation matrix              | [B, N, 3, 3]
            - post_trans (Tensor): augmentation translation matrix          | [B, N, 3]
        Returns:
            - integrated_img (list[Tensor]): Integrated images (Forward, Left, Right)   | [B, N*3, 3, H, W]
        """
        
        theta_left = np.deg2rad(self.angle)
        theta_right = np.deg2rad(-self.angle)
        
        left_img, left_mask, left_depth = self.proj_pcd2plane(lidar_pcd=lidar_pcd, K=intrinsic, rots=rots, trans=trans, post_rots=post_rots, post_trans=post_trans, theta=theta_left)
        right_img, right_mask, right_depth = self.proj_pcd2plane(lidar_pcd=lidar_pcd, K=intrinsic, rots=rots, trans=trans, post_rots=post_rots, post_trans=post_trans, theta=theta_right)
        
        left_view_img = self.sample_image(left_img, left_depth, lidar_pcd_rgb, left_mask)
        right_view_img = self.sample_image(right_img, right_depth, lidar_pcd_rgb, right_mask)
        
        integrated_img = torch.cat((front_image, left_view_img, right_view_img), dim=1)
        
        return integrated_img