from typing import Tuple

from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseTransform

__all__ = ["LSSTransform"]


@VTRANSFORMS.register_module()
class LSSTransform(BaseTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, 1)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x):
        B, N, C, fH, fW = x.shape

        x = x.view(B * N, C, fH, fW)

        x = self.depthnet(x)
        depth = x[:, : self.D].softmax(dim=1)
        ###
        import numpy as np
        # np.savetxt("assets/encoder.camera.depth.output.txt", depth.clone().cpu().numpy().reshape(-1))
        # np.savetxt("assets/encoder.camera.feats.output.txt", x[:, self.D : (self.D + self.C)].permute(0, 2, 3, 1) .clone().cpu().numpy().reshape(-1))
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2) # 原来的代码
        # import torch
        # depth = torch.tensor(np.loadtxt("assets/encoder.camera.depth.output.cpp.txt").reshape(6, 118, 32, 88),
        #                  dtype=torch.float32).to(x.device)
        # print(depth.shape)
        # feats = torch.tensor(np.loadtxt("assets/encoder.camera.feats.output.cpp.txt").reshape(6, 32, 88, 80),
        #                  dtype=torch.float32).to(x.device).permute(0, 3, 1, 2)
        # print(feats.shape)
        # x = depth.unsqueeze(1) * feats.unsqueeze(2)

        ###
        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs)
        import numpy as np
        # np.savetxt("assets/vtransform.downsample.input.txt", x.clone().cpu().numpy().reshape(-1))
        # import torch
        # x = torch.tensor(np.loadtxt("assets/vtransform.downsample.input.cpp.txt").reshape(1, 80, 256, 256),
        #                  dtype=torch.float32).to(x.device)
        x = self.downsample(x)
        # np.savetxt("assets/vtransform.downsample.output.txt", x.clone().cpu().numpy().reshape(-1))
        return x
