 

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding

@HEADS.register_module()
class VoxFormerHead(nn.Module):
    def __init__(
        self,
        *args,
        volume_h,
        volume_w,
        volume_z,
        data_config,
        point_cloud_range,
        embed_dims,
        cross_transformer,
        self_transformer,
        positional_encoding,
        mlp_prior=False,
        **kwargs
    ):
        super().__init__()
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.embed_dims = embed_dims
        
        self.data_config = data_config
        self.point_cloud_range = point_cloud_range
        self.volume_embed = nn.Embedding((self.volume_h) * (self.volume_w) * (self.volume_z), self.embed_dims)
        # self.voxelize = Voxelization(point_cloud_range=point_cloud_range, spatial_shape=np.array([volume_h, volume_w, volume_z]))
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.cross_transformer = build_transformer(cross_transformer)
        self.self_transformer = build_transformer(self_transformer)

        image_grid = self.create_grid()
        self.register_buffer('image_grid', image_grid)
        vox_coords, ref_3d = self.get_voxel_indices()
        self.register_buffer('vox_coords', vox_coords)
        self.register_buffer('ref_3d', ref_3d)

        if mlp_prior:
            self.mlp_prior = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dims//2, self.embed_dims)
            )
        else:
            self.mlp_prior = None
            self.mask_embed = nn.Embedding(1, self.embed_dims)

    def get_voxel_indices(self):
        xv, yv, zv = torch.meshgrid(
            torch.arange(self.volume_h), torch.arange(self.volume_w),torch.arange(self.volume_z), 
            indexing='ij')
        
        idx = torch.arange(self.volume_h * self.volume_w * self.volume_z)
        vox_coords = torch.cat([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)

        ref_3d = torch.cat(
            [(xv.reshape(-1, 1) + 0.5) / self.volume_h, 
             (yv.reshape(-1, 1) + 0.5) / self.volume_w, 
             (zv.reshape(-1, 1) + 0.5) / self.volume_z], dim=-1
        )

        return vox_coords, ref_3d

    def create_grid(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)

        grid = torch.stack((xs, ys), 1)
        return nn.Parameter(grid, requires_grad=False)
    
    def forward(self, mlvl_feats, proposal, cam_params, lss_volume=None, img_metas=None,  **kwargs):
        """ 
        description:
            Forward funtion.
        params:
            - mlvl_feats (tuple[Tensor]): Features from the upstream network    | [B, N, C, H, W]
            - proposal (Tensor): Proposed voxel index on voxel grid             | [B, N, H, W, Z]
            - cam_params (list): Cam params {rot, trans, intrins, post_rot, bda-rot}
            - lss_volume (tensor): LSS volume which reflect depth info          | [B, C, H, W, Z]
            - img_metas(dict): meta data of img {pc_range, occ_size, sequence, frame_id, raw_img, stereo_depth, focal_length
                                                baseline, img_shape, gt_depths}
            - kwargs['mlvl_dpt_dists']: depth distribution                      | [B, depth_bin, H', W']
        returns:
            - vox_feats_diff (Tensor): voxel features after all transformer     | [B, C, H, W, Z]
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype, device = mlvl_feats[0].dtype, mlvl_feats[0].device        

        # Add embedding and lss volume feature
        volume_queries = self.volume_embed.weight.to(dtype)     # volume_queries: [H*W*Z, C]
        if lss_volume is not None:
            # Todo: support batch size > 1
            assert lss_volume.shape[0] == 1

            lss_volume_flatten = lss_volume.flatten(2).squeeze(0).permute(1, 0) # lss_volume_flatten: [H*W*Z, C]
            volume_queries = volume_queries + lss_volume_flatten                # volume_queries: [H*W*Z, C]

        if proposal.sum() < 2:
            proposal = torch.ones_like(proposal)    # proposal: [B, N, H, W, Z]
        
        # Generate bev postional embeddings for cross and self attention
        bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=volume_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]
        bev_pos_self_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=volume_queries.device).to(dtype)).to(dtype) # [1, dim, 128*4, 128*4]

        vox_coords, ref_3d = self.vox_coords.clone(), self.ref_3d.clone()   # vox_coords: [N, 4] / ref_3d: [N, 3]

        unmasked_idx = torch.nonzero(proposal.reshape(-1) > 0).view(-1)     # [N]
        masked_idx = torch.nonzero(proposal.reshape(-1) == 0).view(-1)      # [N]
        
        # Compute seed features of query proposals by deformable cross attention
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats,                 # mlvl_feats: Features from the upstream network    | [B, N, C, H, W]
            volume_queries,             # volume_queries: embedding + LSS volume flatten    | [N, C]
            self.volume_h,              # self.volume_h: voxel height                       | scalar
            self.volume_w,              # self.volume_w: voxel width                        | scalar
            ref_3d=ref_3d,              # ref_3d: 3D reference points with normalization    | [N, 3]
            vox_coords=vox_coords,      # vox_coords: voxel index                           | [N, 4]
            unmasked_idx=unmasked_idx,  # unmasked_idx: voxel index which is unmasked       | [N]
            grid_length=None,           # 
            bev_pos=bev_pos_cross_attn, # bev_pos_cross_attn: positional encoding           | [B, C, 128*4, 128*4]
            img_metas=img_metas,        # img_metas: meta data of image                     | 
            prev_bev=None,              #
            cam_params=cam_params,      # cam_params: camera parameters
            **kwargs                    # kwargs['mlvl_dpt_dists] (list) depth distribution | [B, N, depth_bin, H', W']
        )

        # Process visible voxels
        vox_feats = torch.empty((self.volume_h, self.volume_w, self.volume_z, self.embed_dims), device=volume_queries.device)   # vox_feats: [H, W, Z, C]
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)  # vox_feats_flatten: [H*W*Z, C]
        vox_feats_flatten[vox_coords[unmasked_idx, 3], :] = seed_feats[0]       
        
        # Process invisible voxels
        if self.mlp_prior is None:
            vox_feats_flatten[vox_coords[masked_idx, 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(masked_idx.shape[0], self.embed_dims).to(dtype)
        else:
            vox_feats_flatten[vox_coords[masked_idx, 3], :] = self.mlp_prior(lss_volume_flatten[masked_idx, :])
        
        # feature propagation
        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            mlvl_feats,                 # mlvl_feats: Features from the upstream network    | [B, N, C, H, W]
            vox_feats_flatten,          # vox_feats_flatten: voxel after cross_transformer  | [H*W*Z, C]
            512,                        # bev_h
            512,                        # bev_w
            ref_3d=ref_3d,              # ref_3d: 3D reference points which mean voxel point| [H*W*Z, 3]
            vox_coords=vox_coords,      # vox_coords: voxel index                           | [H*W*Z, 4]
            unmasked_idx=unmasked_idx,  # unmasked_idx: voxel index which is unmasked       | [N]
            grid_length=None,           # 
            bev_pos=bev_pos_self_attn,  # bev_pos_self_attn: positional encoding            | [B, C, 128*4, 128*4]
            img_metas=img_metas,        # img_metas: meta data of image                     |
            prev_bev=None,              #
            cam_params=cam_params,      # cam_params: camera parameters
            **kwargs                    # kwargs['mlvl_dpt_dists] (list) depth distribution | [B, N, depth_bin, H', W']
        )

        vox_feats_diff = vox_feats_diff.reshape(self.volume_h, self.volume_w, self.volume_z, self.embed_dims)   # vox_feats_diff: [H, W, Z, C]
        vox_feats_diff = vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0)    # vox_feats_diff: [B, C, H, W, Z]

        return vox_feats_diff