# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# -----disp2seg
def compute_D(points, norm):
    """
    inputs:
        points            b, 4, H*W
        norm              b, 3, H, W
    outputs:
        D                      b, 1, H, W
    """
    batch_size = points.shape[0]
    norm = norm.reshape(batch_size, 3, -1).permute(0, 2, 1).unsqueeze(2) # b , H*W, 1, 3
    points = points[:, :3, :].permute(0, 2, 1).unsqueeze(3) # b, H*W, 3, 1
    points = points.float()
    D = - norm @ points  # b, H*W

    return D    

# ----- vps
def depth2norm(cam_points, height, width, nei=3):
    pts_3d_map = cam_points[:, :3, :].permute(0,2,1).view(-1, height, width, 3)
    
    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[:,nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[:,nei:-nei, 0:-(2*nei), :]
    pts_3d_map_y0 = pts_3d_map[:,0:-(2*nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[:,nei:-nei, 2*nei:, :]
    pts_3d_map_y1 = pts_3d_map[:,2*nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[:,0:-(2*nei), 0:-(2*nei), :]
    pts_3d_map_x0y1 = pts_3d_map[:,2*nei:, 0:-(2*nei), :]
    pts_3d_map_x1y0 = pts_3d_map[:,0:-(2*nei), 2*nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[:,2*nei:, 2*nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    diff_x0 = diff_x0.reshape(-1, 3)
    diff_y0 = diff_y0.reshape(-1, 3)
    diff_x1 = diff_x1.reshape(-1, 3)
    diff_y1 = diff_y1.reshape(-1, 3)
    diff_x0y0 = diff_x0y0.reshape(-1, 3)
    diff_x0y1 = diff_x0y1.reshape(-1, 3)
    diff_x1y0 = diff_x1y0.reshape(-1, 3)
    diff_x1y1 = diff_x1y1.reshape(-1, 3)

    ## calculate normal by cross product of two vectors
    normals0 = torch.cross(diff_x1, diff_y1)
    normals1 =  torch.cross(diff_x0, diff_y0)
    normals2 = torch.cross(diff_x0y1, diff_x0y0)
    normals3 = torch.cross(diff_x1y0, diff_x1y1)

    normal_vector = normals0 + normals1 + normals2 + normals3
    normal_vectorl2 = torch.norm(normal_vector, p=2, dim = 1)
    normal_vector = torch.div(normal_vector.permute(1,0), normal_vectorl2)
    normal_vector = normal_vector.permute(1,0).view(pts_3d_map_ctr.shape).permute(0,3,1,2)
    normal_map = F.pad( normal_vector, (0,2*nei,2*nei,0),"constant",value=0)
    normal = - F.normalize(normal_map, dim=1, p=2)
    return normal

def compute_mmap(batch_size, norm, vps, H, W, epoch, nei):
    """
    inputs:
        norm                           b,3,H,W       tensor
        vps                               b,6,3             tensor
    outputs:
        mmap                         b,1,H,W       tensor
        mmap_mask           b,1,H,W       tensor
    """
    norm_flatten = norm.permute(0,2,3,1).reshape(batch_size,-1,3) # bxNx3
    vps_6 = vps.repeat((norm_flatten.shape[1], 1, 1, 1)).permute(1,2,0,3) # bx6xNx3
    norm_flatten_6 = norm_flatten.repeat((6, 1, 1, 1)).permute(1,0,2,3) # bx6xNx3
    cos = nn.CosineSimilarity(dim=3, eps=1e-6) 
    cos_sim = cos(vps_6, norm_flatten_6) 
    score, index = torch.max(cos_sim, 1)  
    mmap = index.reshape(batch_size, 1, H, W)
    score_map = score.reshape(batch_size, 1, H, W)
    '''
    When the estimated normal is very close to the given principal direction, \
    NaN with cos greater than 1 will appear, so NaN will be set to 1 here.
    '''
    if torch.any(torch.isnan(score_map)):
        print('nan in mmap compute! set nan = 1')
        torch.nan_to_num(score_map, nan=1)

    # The mask here first comes from the top edge and the right edge in depth2norm.
    mmap_mask = torch.ones_like(mmap).cuda()
    mmap_mask[:, :, :20, :] = 0
    mmap_mask[:, :, -8:, :] = 0
    mmap_mask[:, :, :, :8] = 0
    mmap_mask[:, :, :, -8:] = 0
    '''
    Secondly, an adaptive threshold is used to filter the pixels with too large an Angle deviation,\
    with an initial Angle of about 25 degrees
    '''
    score = 1.633 * epoch + 900
    mmap_mask[1000*score_map < score] = 0
    
    return mmap, mmap_mask, score

def align_smooth_norm(batch_size, mmap, vps, H, W):
    """
    inputs:
        mmap                            b, 1, H, W           tensor
        vps                                  b, 6, 3                 tensor
    outputs:
        smooth_norm            b, 3, H*W            tensor
    """
    mmap_label = mmap.reshape(batch_size, -1)
    mmap_label = mmap_label.repeat(3,1,1).permute(1,0,2)
    vps = vps.permute(0, 2, 1)
    smooth_norm = torch.gather(vps, 2, mmap_label)
    smooth_norm = smooth_norm.reshape(batch_size, 3, H, W)

    return smooth_norm


# ----original layers
def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x, mode='nearest'):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode=mode)


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class SSIM_sparse(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM_sparse, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d((1, 9), 1)
        self.mu_y_pool   = nn.AvgPool2d((1, 9), 1)
        self.sig_x_pool  = nn.AvgPool2d((1, 9), 1)
        self.sig_y_pool  = nn.AvgPool2d((1, 9), 1)
        self.sig_xy_pool = nn.AvgPool2d((1, 9), 1)

        # self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # x = self.refl(x)
        # y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
