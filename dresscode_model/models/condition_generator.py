from enum import Enum
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn

from dresscode_model.models.resblock import ResBlockType, ResBlock

def grid_tensor(depth: int, height: int, width: int, use_cuda: bool = True) -> torch.Tensor:
    grid_x = torch.linspace(-1.0, 1.0, width).view(1, 1, width, 1).expand(depth, height, -1, -1)
    grid_y = torch.linspace(-1.0, 1.0, height).view(1, height, 1, 1).expand(depth, -1, width, -1)

    # Combine the two grids to get a grid of (x,y) coordinates for each pixel.
    if use_cuda :
        grid = torch.cat([grid_x, grid_y], 3).cuda()
    else:
        grid = torch.cat([grid_x, grid_y], 3)
        
    return grid


class ConditionGenerator(nn.Module):
    """ Condition Generator
        HR_VITON: https://arxiv.org/abs/2206.14180

        The flow path is based on ClothFlow:
        https://openaccess.thecvf.com/content_ICCV_2019/papers/Han_ClothFlow_A_Flow-Based_Model_for_Clothed_Person_Generation_ICCV_2019_paper.pdf
    """

    def __init__(
            self,
            cloth_encoder_input_channels: int,
            pose_encoder_input_channels: int,
            output_channels: int,
            n_filters: int = 64,
            norm_layer: nn.Module = nn.BatchNorm2d, # type: ignore
            warp_feature: str = 'T1',
            output_layer_type: str = "relu",
            ):
        
        super(ConditionGenerator, self).__init__()
        self.warp_feature = warp_feature
        self.cloth_encoder_input_channels = cloth_encoder_input_channels
        self.pose_encoder_input_channels = pose_encoder_input_channels
        self.output_channels = output_channels
        self.n_filters = n_filters
        self.norm_layer = norm_layer
        self.warp_feature = warp_feature
        self.output_layer_type = output_layer_type

        self.ClothEncoder = self.get_encoder(cloth_encoder_input_channels, n_filters, norm_layer)
        self.PoseEncoder = self.get_encoder(pose_encoder_input_channels, n_filters, norm_layer)
        self.conv = ResBlock(in_channels=n_filters*4, out_channels=n_filters*8, norm_layer=norm_layer, resblock_type=ResBlockType.SAME)

        self.SegDecoder = self.get_decoder()
        self.output_layer = self.get_out_layer(out_layer_type=output_layer_type)
        self.conv1 = self.get_1x1_conv()
        self.conv2 = self.get_1x1_conv()
        self.flow_conv = self.get_flow_conv()
        self.bottleneck = self.get_bottleneck()


    def get_encoder(self, input_channels: int, n_filters: int, norm_layer: nn.Module) -> nn.Sequential:
        """ Encoder, same for cloth and pose """
        return nn.Sequential(
            ResBlock(in_channels=input_channels, out_channels=n_filters, norm_layer=norm_layer, resblock_type=ResBlockType.DOWN),
            ResBlock(in_channels=n_filters, out_channels=n_filters*2, norm_layer=norm_layer, resblock_type=ResBlockType.DOWN),
            ResBlock(in_channels=n_filters*2, out_channels=n_filters*4, norm_layer=norm_layer, resblock_type=ResBlockType.DOWN),
            ResBlock(in_channels=n_filters*4, out_channels=n_filters*4, norm_layer=norm_layer, resblock_type=ResBlockType.DOWN),
            ResBlock(in_channels=n_filters*4, out_channels=n_filters*4, norm_layer=norm_layer, resblock_type=ResBlockType.DOWN),
        )
    
    def get_decoder(self) -> nn.Sequential:
        """ Decoder path"""
        return nn.Sequential(
            ResBlock(in_channels=self.n_filters*8, out_channels=self.n_filters*4, norm_layer=self.norm_layer, resblock_type=ResBlockType.UP),
            ResBlock(in_channels=self.n_filters*4*3, out_channels=self.n_filters*4, norm_layer=self.norm_layer, resblock_type=ResBlockType.UP),
            ResBlock(in_channels=self.n_filters*4*3, out_channels=self.n_filters*2, norm_layer=self.norm_layer, resblock_type=ResBlockType.UP),
            ResBlock(in_channels=self.n_filters*2*3, out_channels=self.n_filters, norm_layer=self.norm_layer, resblock_type=ResBlockType.UP),
            ResBlock(in_channels=self.n_filters*1*3, out_channels=self.output_channels, norm_layer=self.norm_layer, resblock_type=ResBlockType.UP),
        )
    
    def get_1x1_conv(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels=self.n_filters, out_channels=self.n_filters*4, kernel_size=1, bias=True),
            nn.Conv2d(in_channels=self.n_filters*2, out_channels=self.n_filters*4, kernel_size=1, bias=True),
            nn.Conv2d(in_channels=self.n_filters*4, out_channels=self.n_filters*4, kernel_size=1, bias=True),
            nn.Conv2d(in_channels=self.n_filters*4, out_channels=self.n_filters*4, kernel_size=1, bias=True),
        )
    
    def get_flow_conv(self) -> nn.ModuleList:
        return nn.ModuleList([
            nn.Conv2d(in_channels=self.n_filters*8, out_channels=2, kernel_size=3, bias=True, padding=1),
            nn.Conv2d(in_channels=self.n_filters*8, out_channels=2, kernel_size=3, bias=True, padding=1),
            nn.Conv2d(in_channels=self.n_filters*8, out_channels=2, kernel_size=3, bias=True, padding=1),
            nn.Conv2d(in_channels=self.n_filters*8, out_channels=2, kernel_size=3, bias=True, padding=1),
            nn.Conv2d(in_channels=self.n_filters*8, out_channels=2, kernel_size=3, bias=True, padding=1),
        ])

    def get_bottleneck(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=self.n_filters*4, out_channels=self.n_filters*4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=self.n_filters*4, out_channels=self.n_filters*4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=self.n_filters*2, out_channels=self.n_filters*4, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=self.n_filters, out_channels=self.n_filters, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU()
            ),
        )
    
    def get_out_layer(self, out_layer_type: str = "conv") -> nn.Module:
        """ Output layer """
        if out_layer_type == "relu":
            return ResBlock(
                in_channels = self.n_filters + self.cloth_encoder_input_channels + self.pose_encoder_input_channels,
                out_channels = self.output_channels,
                norm_layer=self.norm_layer,
                resblock_type=ResBlockType.SAME
            )
        elif out_layer_type == "conv":
            return nn.Sequential(
                ResBlock(
                in_channels=self.n_filters + self.cloth_encoder_input_channels + self.pose_encoder_input_channels,
                out_channels=self.n_filters,
                norm_layer=self.norm_layer,
                resblock_type=ResBlockType.SAME
                ),
                nn.Conv2d(self.n_filters, self.output_channels, kernel_size=1, bias=True)
            )
        else:
            raise NotImplementedError(f"out_layer_type {out_layer_type} not implemented")
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x # TODO: implement

    def forward(
            self,
            cloth_encoder_input: torch.Tensor,
            pose_encoder_input: torch.Tensor,
            upsample_method: str = "bilinear",
            ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        
        Ec_list: list[torch.Tensor] = []
        Es_list: list[torch.Tensor] = []
        Ff_list: list[torch.Tensor] = []

        # feature pyramid network
        for i in range(5):
            if i == 0:
                # sequential models are indexable
                Ec_list.append(self.ClothEncoder[0](cloth_encoder_input))
                Es_list.append(self.PoseEncoder[0](pose_encoder_input))
            else:
                Ec_list.append(self.ClothEncoder[0](Ec_list[i-1]))
                Es_list.append(self.PoseEncoder[0](Es_list[i-1]))

        # ClothFlow
        for i in range(5):
            depth, _, height, width = Ec_list[4-i].shape
            grid = grid_tensor(depth, height, width, Ec_list[4-i].is_cuda)

            if i == 0: # before first fusion block
                # concatenating Ec and Es
                Eci = Ec_list[4]
                Esi = Es_list[4]
                E_concat = torch.cat([Eci, Esi,], 1)

                # 3x3 conv
                flow = self.flow_conv[0](E_concat).permute(0, 2, 3, 1)
                # first flow feature
                Ff_list.append(flow)

                # 
                Esf = self.conv(Esi)
                Esf = self.SegDecoder[i](Esf)
            else: # fusion blocks
                # upsample cloth and 
                upscaled_Eci = nn.functional.interpolate(Eci, scale_factor=2, mode=upsample_method) # type: ignore
                Eci = self.conv1[4-i](Ec_list[4-i]) + upscaled_Eci
                
                # this shoudln't be used tbh but okay
                upscaled_Esi = nn.functional.interpolate(Esi, scale_factor=2, mode=upsample_method) # type: ignore
                Esi = self.conv2[4-i](Es_list[4-i]) + upscaled_Esi

                # flow feature
                flow = nn.functional.interpolate(Ff_list[i-1].permute((0, 3, 1, 2)), scale_factor=2, mode=upsample_method).permute((0, 2, 3, 1))
                normalized_flow = torch.cat([
                    flow[:, :, :, 0:1] / ((width / 2 - 1)/ 2.0),
                    flow[:, :, :, 1:2] / ((height / 2 - 1)/ 2.0),
                ], dim=3)
                warped_Eci = nn.functional.grid_sample(Eci, grid + normalized_flow, padding_mode="border")

                # concatenating warped Eci and bottleneck
                E_concat = self.normalize(torch.cat([warped_Eci, self.bottleneck[i-1](Esf)], 1)).permute(0, 2, 3, 1) # type: ignore
                flow = flow + self.flow_conv[i](E_concat)
                Ff_list.append(flow)

                if self.warp_feature == "T1":
                    Esf = self.SegDecoder[i](torch.cat([Esf, Es_list[4-i], warped_Eci], 1)) # type: ignore
                if self.warp_feature == "encoder":
                    warped_Eci = nn.functional.grid_sample(Ec_list[4-i], normalized_flow + grid, padding_mode="border")
                    Esf = self.SegDecoder[i](torch.cat([Esf, Es_list[4-i], warped_Eci], 1)) # type: ignore

        # output
        N, _, iH, iW = cloth_encoder_input.shape
        grid = grid_tensor(N, iH, iW, cloth_encoder_input.is_cuda)
        flow = nn.functional.interpolate(Ff_list[-1].permute((0, 3, 1, 2)), scale_factor=2, mode=upsample_method).permute((0, 2, 3, 1))            
        normalized_flow = torch.cat([
            flow[:, :, :, 0:1] / ((iW / 2 - 1)/ 2.0),
            flow[:, :, :, 1:2] / ((iH / 2 - 1)/ 2.0),
        ], dim=3)
        warped_cloth = nn.functional.grid_sample(cloth_encoder_input, grid + normalized_flow, padding_mode="border")
        Esf = self.output_layer(torch.cat([Esf, pose_encoder_input, warped_cloth], 1)) # type: ignore

        warped_c = warped_cloth[:, :-1, :, :]
        warped_cm = warped_cloth[:, -1:, :, :]

        return Ff_list, Esf, warped_c, warped_cm
