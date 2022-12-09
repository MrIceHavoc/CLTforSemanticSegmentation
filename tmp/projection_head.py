# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class ProjectionHead(BaseDecodeHead):
    """Fully Connected Networks for Projection.

    Args:
    """

    def __init__(self, **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        self.transformer = None

    def init_weights(self, hidden_dim, model_out):
        self.transformer = model_out
        self.transformer.fc = nn.Sequential(
            self.transformer.fc,
            nn.ReLU(inplace=True),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.transformer(x)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        #output = self.cls_seg(output)
        return output
