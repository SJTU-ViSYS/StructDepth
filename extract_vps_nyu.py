import os
import numpy as np
import cv2
import argparse

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import tqdm
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
from lu_vp_detect import VPDetection

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    help='path to nyu data',
                    required=True)
parser.add_argument('--output_dir', type=str,
                    help='where to store extracted segment',
                    required=True)
parser.add_argument('--split', type=str,
                    help='path to a list of images to be detected',
                    required=True)
parser.add_argument('--failed_list', type=str,
                    help='where to store the list of images failed to detect principle direction',
                    required=True)
parser.add_argument('--thresh', type=int,
                    help='the length thresh of detecting lines',
                    required=True)
parser.add_argument('--flip',
                    help='vps detect for flip img',
                    action="store_true")
args = parser.parse_args()

data_path = args.data_path
output_dir = args.output_dir
split = args.split
length_thresh = args.thresh
do_flip = args.flip


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def undistort(image):
    k1 = 2.0796615318809061e-01
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

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = np.array(img.convert('RGB'))
            h, w, c = img.shape
            return img

CROP=16
def extract_vps(filename, index):
    image = pil_loader(filename)
    im_name = filename.split('/')[-1].split('.')[0]
    image = undistort(image)
    h, w, c = image.shape
    image = image[CROP : h-CROP, CROP : w-CROP]
    image = cv2.resize(image, (384,288))
    # flip
    if do_flip:
        image = cv2.flip(image, 1)

    fx = 5.1885790117450188e+02/(640-2*CROP)*384
    fy = 5.1946961112127485e+02/(480-2*CROP)*288
    cx = (3.2558244941119034e+02 - CROP)/(640-2*CROP)*384
    cy = (2.5373616633400465e+02- CROP)/(480-2*CROP)*288
    # flip
    if do_flip:
        cx = 384 - cx
    principal_point = cx, cy
    # about how to choose fx or fy, the author's answer is https://github.com/rayryeng/XiaohuLuVPDetection/issues/4
    focal_length = fx
    seed = 2020
    vpd = VPDetection(length_thresh, principal_point, focal_length, seed)
    vps = vpd.find_vps(image) 
    #assert np.isnan(vps).all() == False, print(vps)
    #vpd.create_debug_VP_image(show_image=False, save_image='vps_vis_25/{}.jpg'.format(index)) 
    vps = np.vstack([vps, -vps]).astype(np.float32)

    return vps

def get_vps(filename, index):
    #vps = extract_vps(filename, index)
    try:
        vps = extract_vps(filename, index)
        if do_flip:
            np.save(os.path.join(output_dir, "flip_nyu_vps_%d.npy"%(index)), vps)
        else:
            np.save(os.path.join(output_dir, "nyu_vps_%d.npy"%(index)), vps)
    
    except:
        f = open(args.failed_list, 'a')
        f.write(filename + '\n')
        f.close()

    return
             
if __name__ == '__main__':
    # multi processing fitting
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    futures = []

    lines = open(split).readlines()
    fps = [os.path.join(args.data_path, line.split()[0]) for line in lines]

    for index, files in enumerate(fps):
        task = partial(get_vps, files, index)
        futures.append(executor.submit(task))

    results = []
    [results.append(future.result()) for future in tqdm.tqdm(futures)]
