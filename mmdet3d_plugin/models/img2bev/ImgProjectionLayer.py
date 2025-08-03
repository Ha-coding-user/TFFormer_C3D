import torch
import torch.nn as nn
import numpy as np
from mmdet.models import HEADS
from mmcv.runner import BaseModule

@HEADS.register_module()
class ImgProjectionLayer(BaseModule):
    def __init__(self,
                 C_in=640,
                 C_out=128,
                 kernel_size=3):
        
        super(ImgProjectionLayer, self).__init__()
        
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        
        self.conv2d_1 = nn.Conv2d(C_in, C_out, kernel_size=kernel_size)
        
    def forward(self, img_feats):
        """
        Descript:
            Image projection layer for match channels
        Args:
            img_feats (Tensor): front, left, right image features   | [B, N*3, C, H', W']
        Returns:
            out (Tensor): Image projection features                 | [B, N*3, C', H', W']
        """
        B, N, D, H, W = img_feats.shape
        img_feats = img_feats.reshape(B*N, D, H, W)         # img_feats: [B*N, C, H, W]
        
        out = self.conv2d_1(img_feats)                      # out: [B*N, C, H, W]
        
        return out.reshape(B, N, -1, H, W)