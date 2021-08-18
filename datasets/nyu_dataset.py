# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import cv2
import random
import numpy as np
import copy
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import time
import h5py
from lu_vp_detect import VPDetection
from datasets.extract_svo_point import PixelSelector
import ctypes
import multiprocessing as mp

CROP = 16

# dataset loader
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = np.array(img.convert('RGB'))
            h, w, c = img.shape
            return img

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    norm = np.array(h5f['norm'])
    norm = np.transpose(norm, (1,2,0))
    valid_mask = np.array(h5f['mask'])

    return rgb, depth, norm, valid_mask

class NYUDataset(data.Dataset):
    def __init__(self,
                data_path,
                filenames,
                height,
                width,
                frame_idxs,
                num_scales,
                is_train=False,
                vps_path = '',
                return_vps=False,
                shared_dict=None):
        super(NYUDataset, self).__init__()

        self.full_res_shape = (480-CROP*2,640-CROP*2) 
        self.K = self._get_intrinsics()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.is_train = is_train
        self.vps_path = vps_path
        self.return_vps = return_vps
        self.pixelselector = PixelSelector()

        self.interp = Image.ANTIALIAS
        if self.is_train: self.loader = pil_loader
        else: self.loader = h5_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):                   
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        if not(self.is_train):
            line = self.filenames[index]
            line = os.path.join(self.data_path, line)
            rgb, depth, norm, valid_mask = self.loader(line) 
            h, w, c = rgb.shape

            rgb = rgb[44: 471, 40: 601, :]
            depth = depth[44: 471, 40: 601]

            rgb = Image.fromarray(rgb)
            depth = Image.fromarray(depth)
            rgb = self.to_tensor(self.resize[0](rgb))
            depth = self.to_tensor(depth)

            K = self.K.copy()
            K[0, :] *= self.width
            K[1, :] *= self.height
            return rgb, depth, K, np.array(np.matrix(K).I)

        inputs = {}

        do_color_aug = random.random() > 0.5
        do_flip = random.random() > 0.5

        line = self.filenames[index].split()
        frame_name = line[0]

        if do_flip:
            if frame_name in ['nyu_depth_v2/bedroom_0086/r-1315251252.453265-3131793654.ppm', 'nyu_depth_v2/living_room_0011/r-1295837629.903855-1295712361.ppm']:
                do_flip = False
                print("-----------")
                print("meet not flip vps frame {}\n do_flip is False".format(frame_name))
                print("-----------")

        line = [os.path.join(self.data_path, l) for l in line]
        for ind, i in enumerate(self.frame_idxs):
            if not i in set([0, -2, -1, 1, 2]):
                continue
            inputs[("color", i, -1)] = self.get_color(line[ind], do_flip)

        # load vps
        if self.return_vps:
            if do_flip:
                vp = self.vps_path + 'flip_nyu_vps_%d.npy'%(index)
            else:
                vp = self.vps_path + 'nyu_vps_%d.npy'%(index)
            inputs[('vps', 0, 0)] = np.load(vp)

        svo_map_resized = np.zeros((self.height, self.width)) # 288 * 384
        img = np.array(inputs[("color", 0, -1)])
        key_points = self.pixelselector.extract_points(img)                                                    
        key_points = key_points.astype(int)
        key_points[:,0] = key_points[:,0] * self.height // 480
        key_points[:,1] = key_points[:,1] * self.width // 640

        # noise 1000 points
        noise_num = 3000 - key_points.shape[0]
        noise_points = np.zeros((noise_num, 2), dtype=np.int32)
        noise_points[:, 0] = np.random.randint(self.height, size=noise_num)
        noise_points[:, 1] = np.random.randint(self.width, size=noise_num)

        svo_map_resized[key_points[:,0], key_points[:,1]] = 1

        inputs['svo_map'] = torch.from_numpy(svo_map_resized.copy())
        svo_map_resized[noise_points[:,0], noise_points[:,1]] = 1
        inputs['svo_map_noise'] = torch.from_numpy(
            svo_map_resized,
        ).float()
        keypoints = np.concatenate((key_points, noise_points), axis=0)
        inputs['dso_points'] = torch.from_numpy(
            keypoints,
        ).float()

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            
            # flip : cx = w - cx
            if do_flip:
                K[0][2] = self.width - K[0][2]
            
            inv_K = np.array(np.matrix(K).I)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            if not i in set([0, -2, -1, 1, 2]):
                continue

            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
    
        return inputs

    def get_color(self, fp, do_flip):
        color = self.loader(fp)
        color = self._undistort(color)

        if do_flip:
            color = cv2.flip(color, 1)

        h, w, c = color.shape
        color = color[CROP: h-CROP, CROP: w-CROP, :]

        return Image.fromarray(color)


    def _get_intrinsics(self):
        h,w = self.full_res_shape
        fx = 5.1885790117450188e+02 / w
        fy = 5.1946961112127485e+02 / h
        cx = (3.2558244941119034e+02 - CROP ) / w
        cy = (2.5373616633400465e+02 - CROP ) / h

        intrinsics =np.array([[fx, 0., cx, 0.], 
                               [0., fy, cy, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics

    def _undistort(self, image):
        k1 =  2.0796615318809061e-01
        k2 = -5.8613825163911781e-01
        p1 = 7.2231363135888329e-04
        p2 = 1.0479627195765181e-03
        k3 = 4.9856986684705107e-01

        fx = 5.1885790117450188e+02
        fy = 5.1946961112127485e+02
        cx = 3.2558244941119034e+02
        cy = 2.5373616633400465e+02

        kmat = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([[k1, k2, p1, p2, k3]])
        image = cv2.undistort(image, kmat, dist)
        return image


class NYUTestDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 return_vps = False,
                output_path = ''
        ):
        super(NYUTestDataset, self).__init__()
        self.full_res_shape = (427, 561) 
        self.K = self._get_intrinsics()
        
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.return_vps = return_vps
        self.output_path = output_path

        self.interp = Image.ANTIALIAS
        self.loader = h5_loader
        self.to_tensor = transforms.ToTensor()

        self.resize = transforms.Resize(
            (self.height, self.width),
            interpolation=self.interp
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        line = self.filenames[index]
        line = os.path.join(self.data_path, line)
        rgb, depth, norm, valid_mask = self.loader(line) 

        rgb = rgb[44: 471, 40: 601, :]

        if self.return_vps:
            img_name = line.split('/')[-1].replace('.h5', '')
            vps = self.get_vps(rgb, img_name)

        depth = depth[44: 471, 40: 601]
        norm = norm[44:471, 40:601, :]
        valid_mask = valid_mask[44:471, 40:601]

        rgb = Image.fromarray(rgb)
        rgb = self.to_tensor(self.resize(rgb))

        depth = self.to_tensor(depth)
        norm = self.to_tensor(norm)
        norm_mask = self.to_tensor(valid_mask)

        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height

        
        if not self.return_vps:
            return rgb, depth, norm, norm_mask, K, np.array(np.matrix(K).I)
        else:
            return rgb, depth, norm, norm_mask, K, np.array(np.matrix(K).I), vps, img_name


    def _get_intrinsics(self):
        h, w = self.full_res_shape
        
        fx = 5.1885790117450188e+02 / w
        fy = 5.1946961112127485e+02 / h
        cx = (3.2558244941119034e+02 - 40) / w
        cy = (2.5373616633400465e+02 - 44) / h

        intrinsics =np.array([[fx, 0., cx, 0.], 
                               [0., fy, cy, 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]], dtype="float32")
        return intrinsics
    
    def get_vps(self, img, img_name):
        color = cv2.resize(img, (self.width, self.height))
        h, w = self.full_res_shape
        fx = 5.1885790117450188e+02/w*self.width
        fy = 5.1946961112127485e+02/h*self.height
        cx = (3.2558244941119034e+02 - 40 )/w*self.width
        cy = (2.5373616633400465e+02 - 44)/h*self.height
        length_thresh = 60
        principal_point = cx, cy
        focal_length = fx
        seed = 2020

        vpd = VPDetection(length_thresh, principal_point, focal_length, seed)
        vps = vpd.find_vps(color)
        
        if not os.path.exists('{}/{}/'.format(self.output_path, img_name)):
            os.makedirs('{}/{}/'.format(self.output_path, img_name))
        vpd.create_debug_VP_image(show_image=False, save_image='{}/{}/rgb_line.png'.format(self.output_path, img_name))
        vps = np.vstack([vps, -vps]).astype(np.float32)
        np.save('{}/{}/vps.npy'.format(self.output_path, img_name), vps)

        return vps
