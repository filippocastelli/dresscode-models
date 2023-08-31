import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import os
from torch.nn.utils import spectral_norm
import numpy as np

import functools


class MultiScaleDiscriminator(nn.Module):
    def __init__(
            self,
            n_input_classes: int,
            n_filters: int = 64,
            n_layers: int = 3,
            downscale: bool = False,
            dropout: bool = False,
            use_sigmoid: bool = False,
            num_scales: int = 3,
            get_inter_feat: bool = False,
            use_spectral_norm: bool = False,
            norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super(MultiScaleDiscriminator, self).__init__()

        self.num_D = num_scales
        self.n_layers = n_layers
        self.get_inter_feat = get_inter_feat
        self.downscale = downscale
        self.n_filters = n_filters
        self.n_input_classes = n_input_classes
        self.dropout = dropout
        self.use_sigmoid = use_sigmoid
        self.use_spectral_norm = use_spectral_norm
        self.norm_layer = norm_layer


        for i in range(num_scales):
            discriminator_block = NLayerDiscriminator(
                n_input_classes=n_input_classes,
                n_filters=n_filters,
                n_layers=n_layers,
                use_sigmoid=use_sigmoid,
                use_spectral_norm=use_spectral_norm,
                get_inter_feat=get_inter_feat,
                dropout=dropout,
                norm_layer=norm_layer,
            )
            if self.get_inter_feat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(discriminator_block, 'model'+str(j)))
                else:
                    setattr(self, 'layer'+str(i), discriminator_block.model)
            
            self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self,
                        model: nn.Module,
                        input: torch.Tensor):
        if self.get_inter_feat:
            result = [input]
            for n in range(len(model)):
                result.append(model[n](result[-1]))
            return result[1:]
        else:
            return model(input)
        
    def forward(self,
                input: torch.Tensor):
        num_D = self.num_D
        result = []
        if self.downscale:
            input_downsampled = self.downsample(input)
        else:
            input_downsampled = input
        
        for i in range(num_D):
            if self.get_inter_feat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def define_discriminator(
        n_input_classes: int,
        n_filters: int = 64,
        n_layers: int = 3,
        downscale: bool = False,
        dropout: bool = False,
        use_sigmoid: bool = False,
        num_scales: int = 3,
        get_inter_feat: bool = False,
        use_spectral_norm: bool = False,
        norm_type: str = "instance",
        gpu_ids: list = None,
):
    norm_layer = get_norm_layer(norm_type=norm_type)
    discriminator = MultiScaleDiscriminator(
        n_input_classes=n_input_classes,
        n_filters=n_filters,
        n_layers=n_layers,
        downscale=downscale,
        dropout=dropout,
        use_sigmoid=use_sigmoid,
        num_scales=num_scales,
        get_inter_feat=get_inter_feat,
        use_spectral_norm=use_spectral_norm,
        norm_layer=norm_layer,
    )
    print(discriminator)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        discriminator.cuda()
    discriminator.apply(weights_init)

class NLayerDiscriminator(nn.Module):
    def __init__(
            self,
            n_input_classes: int,
            n_filters: int = 64,
            n_layers: int = 3,
            norm_layer: nn.Module = nn.BatchNorm2d,
            use_sigmoid: bool = False,
            use_spectral_norm: bool = False,
            get_inter_feat: bool = False,
            dropout: bool = False,
    ):
        
        self.get_inter_feat = get_inter_feat
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.n_input_classes = n_input_classes
        self.dropout = dropout
        self.use_sigmoid = use_sigmoid
        self.use_spectral_norm = use_spectral_norm
        self.norm_layer = norm_layer

        super(NLayerDiscriminator, self).__init__()
        self.spectral_norm = spectral_norm if self.use_spectral_norm else lambda x: x
    
    def get_discriminator(self):

        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1.0) / 2))
        sequence = [[nn.Conv2d(self.n_input_classes, self.n_filters, kernel_size=kernel_size, stride=2, padding=padding), nn.LeakyReLU(0.2, True)]]

        n_filters = self.n_filters
        for n in range(1, self.n_layers):
            n_filters_prev = n_filters
            n_filters = min(n_filters * 2, 512)
            sequence += [[self.spectral_norm(nn.Conv2d(n_filters_prev, n_filters, kernel_size=kernel_size, stride=2, padding=padding)),
                          self.norm_layer(n_filters),
                          nn.LeakyReLU(0.2, True),
                          nn.Dropout(0.5) if self.dropout else None]]
        n_filters_prev = n_filters
        n_filters = min(n_filters * 2, 512)
        sequence += [[
            nn.Conv2d(n_filters_prev, n_filters, kernel_size=kernel_size, stride=1, padding=padding),
            self.norm_layer(n_filters),
            nn.LeakyReLU(0.2, True),
        ]]

        sequence += [[nn.Conv2d(n_filters, 1, kernel_size=kernel_size, stride=1, padding=padding)]]

        if self.use_sigmoid:
            sequence += [[nn.Sigmoid()]]
        
        if self.get_inter_feat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input: torch.Tensor):
        if self.get_inter_feat:
            results = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model'+str(n))
                results.append(model(results[-1]))
            return results[1:]
        else:
            return self.model(input)

        


