import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import BaseModule
from mmdet3d.models import builder
from mmdet.models import HEADS

@HEADS.register_module()
class C3DModule(BaseModule):
    def __init__(self,
                 depth_net,
                 proposal_layer,
                 img_projection_layer,
                 point_feature_alignment,
                 voxel_refinement_layer):
        
        super(C3DModule, self).__init__()
        
        self.depth_net = builder.build_neck(depth_net)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.img_projection_layer = builder.build_head(img_projection_layer)
        self.point_feature_alignment = builder.build_head(point_feature_alignment)
        self.voxel_refinement_layer = builder.build_head(voxel_refinement_layer)
        
    def forward(self, img_enc_feats, stereo_depth, img_inputs, img_metas):
        proposal, lidar_pcd_list = self.proposal_layer(stereo_depth, img_inputs)
        
        proj_img_enc_feats = self.img_projection_layer(img_enc_feats)
        volume = self.point_feature_alignment(proj_img_enc_feats, stereo_depth, lidar_pcd_list)
        
        cur_img_input = img_inputs[-1]
        cur_img_metas = img_metas[-1]
        cur_img_enc_feats = img_enc_feats[:, -1:, ...]
        mlp_input = self.depth_net.get_mlp_input(*cur_img_input[1:7])
        context, depth = self.depth_net([cur_img_enc_feats] + cur_img_input[1:7] + [mlp_input], cur_img_metas)
        
        # voxel refinement
        x = self.voxel_refinement_layer(
            mlvl_feats=[context],
            proposal=proposal,
            cam_params=cur_img_input[1:7],
            lss_volume=volume,
            img_metas=cur_img_metas,
            mlvl_dpt_dists=[depth.unsqueeze(1)]
        )
        
        return x, depth