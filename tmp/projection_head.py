import torch.nn as nn

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
        self.fc = None

    def init_weights(self, in_channels, hidden_dim, model_out):
        self.transformer = model_out
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 4*hidden_dim),
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
        logits = self.transformer(x)
        feats = self.fc(logits)
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        #output = self.cls_seg(output)
        return output
