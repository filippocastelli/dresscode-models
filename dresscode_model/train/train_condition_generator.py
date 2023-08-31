from pathlib import Path
from dataclasses import dataclass, field
import torch
import torch.nn as nn

from torchvision.utils import make_grid as make_image_grid

from dresscode_model.dataset import CpDataset, CPDataLoader
from dresscode_model.models.condition_generator import grid_tensor as mkgrid
from dresscode_model.models.condition_generator import ConditionGenerator
from dresscode_model.models.vgg19 import VGGLoss, Vgg19
from dresscode_model.models.GanLoss import GANLoss
from dresscode_model.models import IoUMetric
from dresscode_model.utils.utils import save_checkpoint, load_checkpoint
from dresscode_model.models.discriminator import get_norm_layer, define_discriminator

@dataclass
class TrainConditionGeneratorOptions:
    name: str = field(default="ConditionGenerator")
    gpu_ids: list = field(default_factory=lambda: [0])
    cuda: bool = field(default=True)
    n_workers: int = field(default=4)
    batch_size: int = field(default=4)
    shuffle: bool = field(default=True)
    pin_memory: bool = field(default=True)
    downscale: bool = field(default=False)
    dropout: bool = field(default=False)
    use_spectral_norm: bool = field(default=False)
    num_scales: int = field(default=3)

    use_amp: bool = field(default=True) # not implemented yet
    dataroot: Path = field(default=Path("./data"))
    datamode: str = field(default="train")
    data_list_path: Path = field(default=Path("./train_pairs.txt"))
    fine_width: int = field(default=256)
    fine_height: int = field(default=192)

    test_visualize: bool = field(default=True)
    tensorboard_dir: Path = field(default=Path("./tensorboard"))
    checkpoint_dir: Path = field(default=Path("./checkpoints"))
    checkpoint: Path = field(default=None)

    n_classes: int = field(default=13)
    n_output_classes: int = field(default=13)

    # network-related stuff
    warp_feature: str = field(default="T1")
    out_layer: str = field(default="relu")




def train(options: TrainConditionGeneratorOptions):

    # dataset and dataloader
    train_dataset = CpDataset(
        root=options.dataroot,
        data_mode=options.datamode,
        data_list=options.data_list_path,
        fine_height=options.fine_height,
        fine_width=options.fine_width,
        semantic_nc=options.n_classes,
    )

    train_dataloader = CPDataLoader(
        dataset=train_dataset,
        shuffle=options.shuffle,
        batch_size=options.batch_size,
        num_workers=options.n_workers,
        pin_memory=options.pin_memory,
    )

    # TODO: visualization

    # Model

    input1_nc = 4 # cloth + cloth-mask
    input2_nc = options.n_classes + 3 # parse_agnostic + densepose
    
    condition_generator = ConditionGenerator(
        cloth_encoder_input_channels=4,
        pose_encoder_input_channels=options.n_classes + 3, # parse_agnostic + densepose
        output_channels=options.n_output_classes,
        n_filters=96,
        norm_layer=nn.BatchNorm2d,
        warp_feature=options.warp_feature,
    )

    discriminator = define_discriminator(
        n_input_classes=input1_nc + input2_nc + options.n_output_classes,
        downscale=options.downscale,
        dropout=options.dropout,
        n_layers=3,
        use_spectral_norm=options.use_spectral_norm,
        num_scales=options.num_scales
    )