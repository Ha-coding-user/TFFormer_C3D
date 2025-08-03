

import numpy as np
import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):

    """
    Attention with both self and cross
    Implements the decoder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, num_cams=3, angle=30, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.num_cams = num_cams
        self.angle = angle

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """
        Descript:
            Get the reference points used in SCA and TSA.
        Args:
            - H, W: spatial shape of bev.
            - Z: hight of pillar.
            - dim: wheter 3D or 2D reference points
            - bs: batch size
            - device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z    # zs: [num_pillar, H, W]
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W     # xs: [num_pillar, H, W]
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H     # ys: [num_pillar, H, W]
            ref_3d = torch.stack((xs, ys, zs), -1)  # ref_3d: [num_pillar, H, W, 3(x, y, z)]
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1) # ref_3d: [num_pillar, 3, H, W] -> [num_pillar, H*W, 3]
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)   # ref_3d: [B, num_pillar, H*W, 3]
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )   # ref_y: [H, W] / ref_x: [H, W]
            ref_y = ref_y.reshape(-1)[None] / H # ref_y: [1, H*W] -> normalize
            ref_x = ref_x.reshape(-1)[None] / W # ref_x: [1, H*W] -> normalize
            ref_2d = torch.stack((ref_x, ref_y), -1)        # ref_2d: [1, H*W, 2]
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)   # ref_2d: [B, H*W, lvl, 2]
            return ref_2d
        
    def calc_turning_mat(self, theta, device):
        turning_extrinsic = torch.Tensor([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        turning_extrinsic = turning_extrinsic.unsqueeze(0).unsqueeze(0).to(device)
        
        return turning_extrinsic

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range, cam_params, img_shape):
        """
        Descript:
            Sampling point from lidar pcd to camera plane
        Args:
            - reference_points (Tensor): 3D ref points                  | [B, num_pillar, H*W, 3(x, y, z)]
            - pc_range (list): point cloud range                        | [depth_min, width_min, height_min, depth_max, width_max, height_max]
            - cam_params (list): cam params                             | {'rot', 'trans', 'intrin', 'post_rot', 'post_trans', 'bda_rot', 'cam2lidar', 'cam2world'}
        Returns:
            - reference_points_img (Tensor): camera reference points    | [num_cam, B, H*W, num_pillar, 2]
            - bev_mask (Tensor): masking which is not valid coord       | [num_cam, B, H*W, num_pillar]
        """
        
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Calculate turning matrix        
        theta_left = np.deg2rad(self.angle)
        theta_right = np.deg2rad(-self.angle)

        forward_E = torch.eye(3).unsqueeze(0).unsqueeze(0).to(reference_points.device)  # [B, N, 3, 3]
        left_E = self.calc_turning_mat(theta_left, reference_points.device)
        right_E = self.calc_turning_mat(theta_right, reference_points.device)
        
        turning_mat = torch.cat((forward_E, left_E, right_E), dim=1)    # turning_mat: [B, num_cam, 3, 3]
        assert turning_mat.size(1) == self.num_cams
        
        # rot: cam2lidar rotation matrix                | [B, N, 3, 3]
        # trans: cam2lidar translation matrix           | [B, N, 3]
        # intrin: intrinsic matrix                      | [B, N, 4, 4]
        # post_rot: augmentation rotation matrix        | [B, N, 3, 3]
        # post_trans: augmentation translation matrix   | [B, N, 3]
        rot, trans, intrin, post_rot, post_trans, bda_rot, _, _ = cam_params

        # normalized reference points -> denormalized(real)
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = reference_points.permute(1, 0, 2, 3)     # D, B, num_query, 3
        D, B, num_query = reference_points.size()[:3]
        
        reference_points = reference_points.view(
            D, B, 1, num_query, 3).repeat(1, 1, self.num_cams, 1, 1).unsqueeze(-1)  # [num_pillar, B, num_cam, num_query, 3, 1]
        
        # lidar -> cam
        rot = rot.unsqueeze(0).unsqueeze(-3).repeat(D, B, self.num_cams, num_query, 1, 1)   # [num_pillar, B, num_cam, num_query, 3, 3]
        trans = trans.unsqueeze(0).unsqueeze(-2).repeat(D, B, self.num_cams, num_query, 1).unsqueeze(-1)
        reference_points_cam = torch.inverse(rot) @ reference_points - trans    # reference_points_cam: [num_pillar, B, num_cam, num_query, 3, 1]
        
        # turning view
        turning_mat = turning_mat.unsqueeze(0).unsqueeze(-3).repeat(D, B, 1, num_query, 1, 1)   # turning_mat: [num_pillar, B, num_cam, num_query, 3, 3]
        reference_points_turning = turning_mat @ reference_points_cam   # reference_points_turning: [num_pillar, B, num_cam, num_query, 3, 1]
        
        # cam -> image plane
        intrin = intrin.unsqueeze(0).unsqueeze(-3).repeat(D, B, self.num_cams, num_query, 1, 1) # intrin: [num_pillar, B, num_cam, num_query, 3, 3]
        reference_points_img = intrin[..., :3, :3] @ reference_points_turning + intrin[..., :3, 3:] # reference_points_img: [num_pillar, B, num_cam, num_query, 3, 1]
        reference_points_img = reference_points_img.squeeze(-1)
        
        eps = 1e-5
        
        bev_mask = (reference_points_img[..., 2:3] > eps)   # bev_mask: [num_pillar, B, num_cam, num_query, 1]
        reference_points_img = reference_points_img[..., 0:2] / torch.maximum(
            reference_points_img[..., 2:3], torch.ones_like(reference_points_img[..., 2:3]) * eps)  # reference_points_img: [num_pillar, B, num_cam, num_query, 2]

        # post processs
        post_rot = post_rot.unsqueeze(0).unsqueeze(-3).repeat(D, B, self.num_cams, num_query, 1, 1) # post_rot: [num_pillar, B, num_cam, num_query, 3, 3]
        post_trans = post_trans.unsqueeze(0).unsqueeze(-2).repeat(D, B, self.num_cams, num_query, 1).unsqueeze(-1)  # post_trans: [num_pillar, B, num_cam, num_query, 3, 1]
        reference_points_img = (post_rot @ reference_points_img + post_trans).squeeze(-1)   # reference_points_img: [num_pillar, B, num_cam, num_query, 3]
        

        reference_points_img[..., 0] /= img_shape[1]
        reference_points_img[..., 1] /= img_shape[0]

        bev_mask = (bev_mask & (reference_points_img[..., 1:2] > 0.0)
                    & (reference_points_img[..., 1:2] < 1.0)
                    & (reference_points_img[..., 0:1] < 1.0)
                    & (reference_points_img[..., 0:1] > 0.0))   # bev_mask: [num_pillar, B, num_cam, num_query, 1]
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            bev_mask = torch.nan_to_num(bev_mask)
        else:
            bev_mask = bev_mask.new_tensor(
                np.nan_to_num(bev_mask.cpu().numpy()))

        reference_points_img = reference_points_img.permute(2, 1, 3, 0, 4)  # reference_points_img: [num_cam, B, num_query, num_pillar, 2]
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)  # bev_mask: [num_cam, B, num_query, num_pillar]

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_img, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """
        Descript:
            Forward function for BEVFormerEncoder
        Args:
            - bev_query (Tensor): Input Bev query                                           | [H*W, B, C]
            - key (Tensor): Input multi-camera features                                     | [N, H'*W", B, C]
            - value (Tensor): Input multi-camera features                                   | [N, H'*W', B, C]
            - args  
            - bev_h (scalar): height of bev
            - bev_w (scalar): width of bev
            - spatial_shapes (tuple): multi-camera features shape                           | = [H', W']
            - level_start_index (Tensor): multi-camera features start index when flatten    | = [0]
            - valid_ratios ()
            - prev_bev (Tensor or None): previous bev features                              |
            - shift (?)
            - kwargs['img_inputs']: camera params                                           | {'rot', 'trans', 'intrin', 'post_rot', 'post_trans', 'bda_rot', 'cam2lidar', 'cam2world'}
        Returns:
            - output (Tensor): output bev features                                          | [B, H*W, C]
        """

        output = bev_query  # output: [H*W, B, C]
        intermediate = []

        ref_3d = self.get_reference_points( # ref_3d: [B, num_pillar, H*W, 3(x, y, z)]
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype)
        ref_2d = self.get_reference_points( # ref_2d: [B, H*W, lvl, 2(x, y)]
            bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)
        
        img_shape = kwargs['img_metas']['img_shape']

        # sampling point by using reference 3d(lidar pcd) and calib_matrix...
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, cam_params=kwargs['img_inputs'], img_shape=img_shape)

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.clone()   # shift_ref_2d: [B, H*W, lvl, 2]
        if isinstance(shift, float):
            shift_ref_2d += shift
        else:
            shift_ref_2d += shift[:, None, None, :] 

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query




from mmcv.cnn.bricks.transformer import build_feedforward_network, build_attention


@TRANSFORMER_LAYER.register_module()
class MM_BEVFormerLayer(MyCustomBaseTransformerLayer):
    """multi-modality fusion layer.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 lidar_cross_attn_layer=None,
                 **kwargs):
        super(MM_BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])
        self.cross_model_weights = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True) 
        if lidar_cross_attn_layer:
            self.lidar_cross_attn_layer = build_attention(lidar_cross_attn_layer)
            # self.cross_model_weights+=1
        else:
            self.lidar_cross_attn_layer = None


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                debug=False,
                depth=None,
                depth_z=None,
                lidar_bev=None,
                radar_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    lidar_bev=lidar_bev,
                    reference_points=ref_2d,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                new_query1 = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    depth=depth,
                    lidar_bev=lidar_bev,
                    depth_z=depth_z,
                    **kwargs)

                if self.lidar_cross_attn_layer:
                    bs = query.size(0)
                    new_query2 = self.lidar_cross_attn_layer(
                        query,
                        lidar_bev,
                        lidar_bev,
                        reference_points=ref_2d[bs:],
                        spatial_shapes=torch.tensor(
                            [[bev_h, bev_w]], device=query.device),
                        level_start_index=torch.tensor([0], device=query.device),
                        )
                query = new_query1 * self.cross_model_weights + (1-self.cross_model_weights) * new_query2
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
