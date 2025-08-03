import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder

@DETECTORS.register_module()
class TFFormer(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        view_transformer_layer,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        depth_loss=True,
        train_cfg=None,
        test_cfg=None
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        
        self.view_transformer_layer = builder.build_head(view_transformer_layer)

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        
        self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.depth_loss = depth_loss

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas, post_img_inputs, post_img_metas):
        """
        params
            - img_input(list): current img inputs / camera params   | info: {'img', 'rot', 'trans', 'intrin', 'post_rot', 'post_trans', 'bda_rot', 'cam2lidar', 'cam2world'}
            - img_metas(dict): current img's meta data              | keys: {'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape', 'gt_depths'}
            - post_img_inputs(list): post imgs inputs / cam paras   | info: {'img', 'rot', 'trans', 'intrin', 'post_rot', 'post_trans', 'bda_rot', 'cam2lidar', 'cam2world'}
            - post_img_metas(dict): post imgs meta data             | keys: {'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape', 'gt_depths'}
            
        returns
            - x (Tensor): 
            - depth (Tensor):
            - bev_feat (Tensor): 
        """
        img_list = []
        with torch.no_grad():
            for post_img_input in post_img_inputs:
                img_list.append(self.image_encoder(post_img_input[0]))
        img_list.append(self.image_encoder(img_inputs[0]))
        
        img_enc_feats = torch.cat(img_list, dim=1)
        
        total_img_inputs = post_img_inputs + [img_inputs]
        total_img_metas = post_img_metas + [img_metas]
        total_stereo_depth = [img_meta['stereo_depth'] for img_meta in total_img_metas]
        
        x, depth = self.view_transformer_layer(img_enc_feats,
                                               total_stereo_depth,
                                               total_img_inputs,
                                               total_img_metas)
        
        return x, depth
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):
        """
        params
            :data_dict - list of input data dictionary (t-l, ..., t)    | keys: {'img_metas', 'img_inputs', 'gt_occ'}
                'img_metas' : meta data             | keys: {'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape', 'gt_depths'}
                'img_inputs': input img / cam_mat   | info: {'img', 'rot', 'trans', 'intrin', 'post_rot', 'post_trans', 'bda_rot', 'cam2lidar', 'cam2world'}
                'gt_occ'    : gt data               | (256, 256, 32)
                
        returns
            :test_output - dictionary of output info    | keys: {'pred', 'gt_occ', 'losses'}
                'pred'  : prediction results    | (256, 256, 32)
                'gt_occ': gt occupancy          | (256, 256, 32)
                'losses': loss dictionary       | keys: {'loss_depth', ???}
        """
        img_inputs = data_dict[-1]['img_inputs']
        img_metas = data_dict[-1]['img_metas']
        
        post_img_inputs = [post_img_input['img_inputs'] for post_img_input in data_dict[:-1]]   # t-l -> t
        post_img_metas = [post_img_input['img_metas'] for post_img_input in data_dict[:-1]]     # t-l -> t
        
        gt_occ = data_dict[-1]['gt_occ']

        img_voxel_feats, depth = self.extract_img_feat(img_inputs, img_metas, post_img_inputs, post_img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        losses = dict()

        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(img_metas['gt_depths'], depth)

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    
    def forward_test(self, data_dict):
        """
        params
            :data_dict - list of input data dictionary (t-l, ..., t)    | keys: {'img_metas', 'img_inputs', 'gt_occ'}
                'img_metas' : meta data             | keys: {'pc_range', 'occ_size', 'sequence', 'frame_id', 'raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape', 'gt_depths'}
                'img_inputs': input img / cam_mat   | info: {'img', 'rot', 'trans', 'intrin', 'post_rot', 'post_trans', 'bda_rot', 'cam2lidar', 'cam2world'}
                'gt_occ'    : gt data               | (256, 256, 32)
                
        returns
            :test_output - dictionary of output info    | keys: {'pred', 'gt_occ'}
                'pred'  : prediction results    | (256, 256, 32)
                'gt_occ': gt occupancy          | (256, 256, 32)
        """
        img_inputs = data_dict[-1]['img_inputs']
        img_metas = data_dict[-1]['img_metas']
        
        post_img_inputs = [post_img_input['img_inputs'] for post_img_input in data_dict[:-1]]   # t-l -> t
        post_img_metas = [post_img_input['img_metas'] for post_img_input in data_dict[:-1]]     # t-l -> t
        
        gt_occ = data_dict[-1]['gt_occ']

        img_voxel_feats, depth = self.extract_img_feat(img_inputs, img_metas, post_img_inputs, post_img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]
        
        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)