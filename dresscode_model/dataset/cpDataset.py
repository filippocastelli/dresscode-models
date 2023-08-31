from pathlib import Path
import json
from typing import List, Tuple, Union, Dict, Any

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
import numpy as np


class CpDataset(data.Dataset):
    """CP-VTON+ dataset.
    """

    def __init__(
            self,
            root: Path,
            data_mode: str = "train",
            data_list: str = "train_pairs.txt",
            fine_height: int = 1024,
            fine_width: int = 768,
            semantic_nc: int = 13,
    ):
        self.root = Path(root)
        self.data_mode = data_mode
        self.data_list = data_list
        self.fine_height = fine_height
        self.fine_width = fine_width
        self.semantic_nc = semantic_nc

        self.data_path = self.root.joinpath(self.data_mode)
        assert self.data_path.exists(), f"{self.data_path} does not exist."

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.im_names_lst: List[str] = []
        self.c_names_lst: List[str] = []

        with open(self.root.joinpath(self.data_list), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                self.im_names_lst.append(im_name)
                self.c_names_lst.append(c_name)
        
        self.c_names: Dict[str, List[str]] = {
            "paired": self.c_names_lst,
            "unpaired": self.c_names_lst,
        }
        self.c_keys = ["paired"]

    def name(self) -> str:
        return "CPDataset"
    
    def get_agnostic(
            self,
            im: Image,
            im_parse: Image,
            pose_data: np.ndarray
    ) -> Image:
        
        parse_array = np.array(im_parse)

        filtered = lambda x: np.where(parse_array == x, 1, 0)
        parse_head = np.sum([filtered(i) for i in [4, 13]], axis=0)
        parse_lower = np.sum([filtered(i) for i in [9, 12, 16, 17, 18, 19]], axis=0)

        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        r = int(length_a / 16) + 1

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            # mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
            mask_arm = Image.new('L', (768, 1024), 'white')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
            for i in pose_ids[1:]:
                if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
            mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
        return agnostic
    
    def __getitem__(self, index: int) -> dict:
        im_name = self.im_names_lst[index]
        im_name = 'image/' + im_name

        cloth_name = {}
        cloth_dict = {}
        cloth_mask_dict = {}

        for key in self.c_keys:
            cloth_name[key] = self.c_names[key][index]

            cloth_img_path = self.data_path / "cloth" / cloth_name[key]
            cloth_img = Image.open(cloth_img_path).convert("RGB")
            cloth_img = transforms.Resize(self.fine_width, interpolation=2)(cloth_img)
            cloth_img = self.transform(cloth_img)
            cloth_dict[key] = cloth_img

            # binary image
            cloth_mask_path = self.data_path / "cloth-mask" / cloth_name[key]
            cloth_mask = Image.open(cloth_mask_path)
            cloth_mask = transforms.Resize(self.fine_width, interpolation=0)(cloth_mask)
            cloth_mask = (np.array(cloth_mask) >= 128).astype(np.float32)
            cloth_mask = torch.from_numpy(cloth_mask).unsqueeze(0)
            cloth_mask_dict[key] = cloth_mask
        
        # person image
        im_pil_path = self.data_path / im_name
        im_pil_large = Image.open(im_pil_path)
        im_pil = transforms.Resize(self.fine_width, interpolation=2)(im_pil_large)
        im = self.transform(im_pil)

        # load parsing images
        parse_name = im_name.replace("image", "image-parse-v3").replace(".jpg", ".png")
        im_parse_pil_large = Image.open(self.data_path.joinpath(parse_name))
        im_parse_pil = transforms.Resize(self.fine_width, interpolation=0)(im_parse_pil_large)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()

        # parse map
        labels = {
            0:  ['background',  [0, 10]],
            1:  ['hair',        [1, 2]],
            2:  ['face',        [4, 13]],
            3:  ['upper',       [5, 6, 7]],
            4:  ['bottom',      [9, 12]],
            5:  ['left_arm',    [14]],
            6:  ['right_arm',   [15]],
            7:  ['left_leg',    [16]],
            8:  ['right_leg',   [17]],
            9:  ['left_shoe',   [18]],
            10: ['right_shoe',  [19]],
            11: ['socks',       [8]],
            12: ['noise',       [3, 11]]
        }

        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        # load image-parse-agnostic
        image_parse_agnostic = Image.open(self.data_path.joinpath(parse_name.replace('image-parse-v3', 'image-parse-agnostic-v3.2')))
        image_parse_agnostic = transforms.Resize(self.fine_width, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        image_parse_agnostic = self.transform(image_parse_agnostic.convert('RGB'))

        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]


        # parse cloth & parse cloth mask
        parse_cloth_mask = new_parse_map[3:4]
        im_c = im * parse_cloth_mask + (1 - parse_cloth_mask)

        # load pose points 
        pose_name = im_name.replace("image", "openpose_img").replace(".jpg", "_rendered.png")
        pose_map = Image.open(self.data_path.joinpath(pose_name))
        pose_map = transforms.Resize(self.fine_width, interpolation=2)(pose_map)
        pose_map = self.transform(pose_map)

        # pose name
        pose_name = im_name.replace("image", "openpose_json").replace(".jpg", "_keypoints.json")
        with open(self.data_path.joinpath(pose_name), "r") as f:
            pose_label = json.load(f)
            pose_data = pose_label["people"][0]["pose_keypoints_2d"]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]

        # load densepose
        densepose_name = im_name.replace("image", "image-densepose")
        densepose_map = Image.open(self.data_path.joinpath(densepose_name))
        densepose_map = transforms.Resize(self.fine_width, interpolation=2)(densepose_map)
        densepose_map = self.transform(densepose_map)

        # agnostic
        agnostic = self.get_agnostic(im_pil_large, im_parse_pil_large, pose_data)
        agnostic = transforms.Resize(self.fine_width, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)

        result = {
            'c_name':   cloth_dict,     # for visualization
            'im_name':  im_name,    # for visualization or ground truth
            # intput 1 (clothfloww)
            'cloth':    cloth_img,          # for input
            'cloth_mask':     cloth_mask,   # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'pose': pose_map,       # for conditioning
            # generator input
            'agnostic' : agnostic,
            # GT
            'parse_onehot' : parse_onehot,  # Cross Entropy
            'parse': new_parse_map, # GAN Loss real
            'pcm': parse_cloth_mask,             # L1 Loss & vis
            'parse_cloth': im_c,    # VGG Loss & vis
            # visualization & GT
            'image':    im,         # for visualization
            }

        return result

    def __len__(self):
        return len(self.im_names_lst)
    
class CPDataLoader:
    def __init__(
            self,
            dataset: CpDataset,
            shuffle: bool = True,
            batch_size: int = 4,
            num_workers: int = 4,
            pin_memory: bool = True,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset = dataset

        if self.shuffle:
            self.sampler = data.RandomSampler(dataset)
        else:
            self.sampler = None
        
        self.data_loader = data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=(self.sampler is None),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            sampler=self.sampler,
        )
        self.data_iter = self.data_loader.__iter__()

    def __len__(self):
        return len(self.data_loader)
    
    def next_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = next(self.data_iter)
        return batch
    

