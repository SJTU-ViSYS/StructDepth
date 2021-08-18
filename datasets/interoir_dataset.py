# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image  # using pillow-simd for increased speed
import random
import torch
import torch.utils.data as data
from torchvision import transforms
from lu_vp_detect import VPDetection

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))

class InteriorTestDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
        ):
        super(InteriorTestDataset, self).__init__()
        
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.interp = Image.ANTIALIAS

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(
            (self.height, self.width),
            interpolation=self.interp
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rgb = self.filenames[index]
        rgb = os.path.join(self.data_path, rgb)
        depth = rgb.replace('cam0', 'depth0')
        
        rgb = self.loader(rgb)
        depth = cv2.imread(depth, -1)/1000.
       
        rgb = Image.fromarray(rgb)

        rgb = self.to_tensor(self.resize(rgb))
        depth = self.to_tensor(depth)
        
        return rgb, depth