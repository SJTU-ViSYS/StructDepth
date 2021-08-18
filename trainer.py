# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from re import S
import torch
import datasets
import numpy as np
import time
import weakref
import math


import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import json
import torchvision
from utils import *
from layers import *
import datasets
import networks
import random
from skimage.segmentation import all_felzenszwalb as felz_seg

# seed
torch.manual_seed(123)
np.random.seed(123)
random.seed(123)

class Trainer:
    def __init__(self, options):

        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
            
        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        # depth encoder
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        # depth decoder
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        # pose encoder
        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        # pose decoder
        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        # optimizer
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        # loading weights
        if self.opt.load_weights_folder is not None:
            self.load_model()
        
        # dataset
        datasets_dict = {"nyu": datasets.NYUDataset}
        self.dataset = datasets_dict[self.opt.train_dataset]
        if self.opt.train_dataset == "nyu":
            train_filenames = readlines(self.opt.train_split)
            val_filenames = readlines(self.opt.val_split)
            train_dataset = self.dataset(
                    self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
                    self.opt.frame_ids, 1, is_train=True, 
                    vps_path=self.opt.vps_path,
                    return_vps=self.opt.using_disp2seg or self.opt.using_normloss)
            self.train_loader = DataLoader(
                    train_dataset, self.opt.batch_size, True,
                    num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
            val_dataset = self.dataset(self.opt.val_path, val_filenames,
                    self.opt.height, self.opt.width,[0], 1, is_train=False)
            self.val_dataloader = DataLoader(val_dataset, 1, shuffle=False, num_workers=self.opt.num_workers)
            self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log10", "da/a1", "da/a2", "da/a3"]
        else:
            print('No implementation for other dataset. Please check options.')
            exit()

        num_train_samples = len(train_filenames)
        self.steps_for_each_epoch = num_train_samples // self.opt.batch_size
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        # process modules setting
        self.ssim_sparse = SSIM_sparse()
        self.ssim_sparse.to(self.device)

        self.ssim = SSIM()
        self.ssim.to(self.device)
        
        self.pdist = nn.PairwiseDistance(p=2)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        print("Training is using:\n  ", self.device)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using frames: \n  ", self.opt.frame_ids_to_train)
        print("Using train split:  ", self.opt.train_split)
        print("There are {:d} training items and {:d} validation items\n".format(len(train_dataset), -1))
        print("Using norm loss:  ", self.opt.using_normloss)
        print("Using planar loss:  ", self.opt.using_disp2seg)
        print("{} for normloss {} for planar loss".format(self.opt.lambda_norm_reg, self.opt.lambda_planar_reg))
        if self.opt.using_normloss or self.opt.using_disp2seg:
            print("vps_path: ", self.opt.vps_path)
        print("start epoch: ", self.opt.start_epoch)
        print("load weights folder: \n", self.opt.load_weights_folder)
        
        if self.opt.start_epoch == 0:
            assert os.path.exists(self.log_path) == False, print('start epoch from 0 but log path conflict \n check log path')
        
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        
        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        # self.step = 0
        # from strat_epoch
        self.step = self.opt.start_epoch * self.steps_for_each_epoch
        print("Training start from {} epoch {} step".format(self.opt.start_epoch, self.step))
        with open(os.path.join(self.log_path, 'eval_res_for_each_epoch.txt'), 'a') as f:
            f.write('\n\n######program start######')
            f.write("\nTraining start from {} epoch {} step".format(self.opt.start_epoch, self.step))
        self.start_time = time.time()
        if self.opt.train_dataset == "nyu":
            self.val()
        elif self.opt.train_dataset == "kitti":
            self.val_kitti()
        for self.epoch in range(self.opt.start_epoch, self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            if self.opt.train_dataset == "nyu":
                self.val()
            elif self.opt.train_dataset == "kitti":
                self.val_kitti()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        self.mmap_thresh_save_flag = True

        self.set_train()
        for param in self.model_optimizer.param_groups:
            with open(os.path.join(self.log_path, 'eval_res_for_each_epoch.txt'), 'a') as f:
                f.write('\nepoch {} time {}'.format(self.epoch, time.asctime( time.localtime(time.time()) )))
                f.write('\nlr {}\n'.format(param["lr"]))
            print("lr:", param["lr"])

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time
            self.log_time(batch_idx, duration, losses)

            if self.step % self.opt.log_frequency == 0:
                self.log("train", inputs, outputs, losses)

            for items in outputs.items():
                del items

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
           Inputs -> dict consists of :
            vps at 0 scale(if self.opt.using_disp2seg or self.opt.using_normloss)
            K/inv_K
            color and color augmented versions   
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        # get depth
        for i in [0]:
            features = self.models["encoder"](inputs[("color_aug", i, 0)])
            output = self.models["depth"](features)
            output = {(disp, i, scale): output[(disp, scale)] for (disp, scale) in output.keys()}
            outputs.update(output)
        # get pose
        outputs.update(self.predict_poses(inputs, features))
        # get planar depth and sparse pred
        self.generate_sparse_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids_to_train}

        for f_i in [-2, -1, 0, 1] if len(self.opt.frame_ids_to_train) == 5 else [-1, 0]:
            # To maintain ordering we always pass frames in temporal order
            pose_inputs = [pose_feats[f_i], pose_feats[f_i + 1]]
            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", f_i, f_i + 1)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=False)

        if len(self.opt.frame_ids_to_train) == 5: 
            outputs[("cam_T_cam", 0, 2)] = outputs[("cam_T_cam", 0, 1)] @ outputs[("cam_T_cam", 1, 2)]
            outputs[("cam_T_cam", -2, 0)] = outputs[("cam_T_cam", -2, -1)] @ outputs[("cam_T_cam", -1, 0)]
            outputs[("cam_T_cam", 0, -2)] = inv_SE3(outputs[("cam_T_cam", -2, 0)])

        outputs[("cam_T_cam", 0, -1)] = inv_SE3(outputs[("cam_T_cam", -1, 0)])

        return outputs
    
    def generate_sparse_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", 0, scale)]
            disp = F.interpolate(
                      disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth
            
            
            if self.opt.using_normloss or self.opt.using_disp2seg:
                cam_points = self.backproject_depth[source_scale](                                   
                    depth, inputs[("inv_K", source_scale)])
                outputs[('cam_points', source_scale)] = cam_points
                self.compute_smooth_norm(inputs, outputs)
                
                if self.opt.using_disp2seg: 
                    self.generate_planar_depth(inputs, outputs, 0, scale)
            
            # sample depth for dso points                                                
            dso_points = inputs['dso_points']
            y0 = dso_points[:, :, 0]
            x0 = dso_points[:, :, 1]
            dso_points = torch.cat((x0.unsqueeze(2), y0.unsqueeze(2)), dim=2)

            flat = (x0 + y0 * self.opt.width).long()
            dso_depth = torch.gather(depth.view(self.opt.batch_size, -1), 1, flat)

            # generate pattern
            meshgrid = np.meshgrid([-2, 0, 2],[-2, 0, 2], indexing='xy')
            meshgrid = np.stack(meshgrid, axis=0).astype(np.float32)
            meshgrid = torch.from_numpy(meshgrid).to(dso_points.device).permute(1, 2, 0).view(1, 1, 9, 2)
            dso_points = dso_points.unsqueeze(2) + meshgrid
            dso_points = dso_points.reshape(self.opt.batch_size, -1, 2)
            dso_depth = dso_depth.view(self.opt.batch_size, -1, 1).expand(-1, -1, 9).reshape(self.opt.batch_size, 1, -1)

            # convert to point cloud
            xy1 = torch.cat((dso_points, torch.ones_like(dso_points[:, :, :1])), dim=2)
            xy1 = xy1.permute(0, 2, 1)
            cam_points = (inputs[("inv_K", source_scale)][:, :3, :3] @ xy1) * dso_depth
            points = torch.cat((cam_points, torch.ones_like(cam_points[:, :1, :])), dim=1)
            outputs[("cam_T_cam", 0, 0)] = torch.eye(4).view(1, 4, 4).expand(self.opt.batch_size, 4, 4).cuda()

            for _, frame_id in enumerate(self.opt.frame_ids_to_train):
                T = outputs[("cam_T_cam", 0, frame_id)]

                # projects to different frames
                P = torch.matmul(inputs[("K", source_scale)], T)[:, :3, :]
                cam_points = torch.matmul(P, points)
                pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
                pix_coords = pix_coords.view(self.opt.batch_size, 2, -1, 9)
                pix_coords = pix_coords.permute(0, 2, 3, 1)
                pix_coords[..., 0] /= self.opt.width - 1
                pix_coords[..., 1] /= self.opt.height - 1
                pix_coords = (pix_coords - 0.5) * 2

                # save mask
                valid = (pix_coords[..., 0] > -1.) & (pix_coords[..., 0] < 1.) & (pix_coords[..., 1] > -1.) & (
                            pix_coords[..., 1] < 1.)
                outputs[("dso_mask", frame_id, scale)] = valid.unsqueeze(1).float()

                # sample patch from color images
                outputs[("dso_color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border")

    def generate_planar_depth(self, inputs, outputs, frame_id, scale):  
        source_scale = 0
        cam_points = outputs[('cam_points', source_scale)]
                    
        if self.opt.using_disp2seg:
            self.compute_seg(inputs, outputs)
            segment = outputs[("disp2seg", 0, source_scale)].unsqueeze(1)
        else:
            segment = inputs[('segment', frame_id, 0)].long()

        max_num = segment.max().item() + 1

        sum_points = torch.zeros((self.opt.batch_size, max_num, 3)).to(self.device)
        area = torch.zeros((self.opt.batch_size, max_num)).to(self.device)
        for channel in range(3):
            points_channel = sum_points[:, :, channel]
            points_channel = points_channel.reshape(self.opt.batch_size, -1)
            points_channel.scatter_add_(1, segment.view(self.opt.batch_size, -1),
                                        cam_points[:, channel, ...].view(self.opt.batch_size, -1))

        area.scatter_add_(1, segment.view(self.opt.batch_size, -1),
                          torch.ones_like(outputs[("depth", 0, source_scale)]).view(self.opt.batch_size, -1))

        # X^T X
        cam_points_tmp = cam_points[:, :3, :]
        x_T_dot_x = (cam_points_tmp.unsqueeze(1) * cam_points_tmp.unsqueeze(2))
        x_T_dot_x = x_T_dot_x.view(self.opt.batch_size, 9, -1)
        X_T_dot_X = torch.zeros((self.opt.batch_size, max_num, 9)).cuda()
        for channel in range(9):
            points_channel = X_T_dot_X[:, :, channel]
            points_channel = points_channel.reshape(self.opt.batch_size, -1)
            points_channel.scatter_add_(1, segment.view(self.opt.batch_size, -1),
                                        x_T_dot_x[:, channel, ...].view(self.opt.batch_size, -1))
        xTx = X_T_dot_X.view(self.opt.batch_size, max_num, 3, 3)

        # take inverse
        xTx_inv = mat_3x3_inv(xTx.view(-1, 3, 3) + 0.01*torch.eye(3).view(1,3,3).expand(self.opt.batch_size*max_num, 3, 3).cuda())
        xTx_inv = xTx_inv.view(xTx.shape)
        xTx_inv_xT = torch.matmul(xTx_inv, sum_points.unsqueeze(3))
        plane_parameters = xTx_inv_xT.squeeze(3)

        # generate mask for segment with area larger than 200
        planar_area_thresh = self.opt.planar_thresh
        valid_mask = ( area > planar_area_thresh ).float()
        planar_mask = torch.gather(valid_mask, 1, segment.view(self.opt.batch_size, -1))
        planar_mask = planar_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)
        # the mask comes from the top edge and the right edge in depth2norm.
        # nei = self.opt.d2n_nei
        # planar_mask[:,:,:2*nei,:] = 0
        # planar_mask[:,:,:,-2*nei:] = 0
        planar_mask[:, :, :8, :] = 0
        planar_mask[:, :, -8:, :] = 0
        planar_mask[:, :, :, :8] = 0
        planar_mask[:, :, :, -8:] = 0
        outputs[("planar_mask", frame_id, scale)] = planar_mask
        
        # segment unpooling
        unpooled_parameters = []
        for channel in range(3):
            pooled_parameters_channel = plane_parameters[:, :, channel]
            pooled_parameters_channel = pooled_parameters_channel.reshape(self.opt.batch_size, -1)
            unpooled_parameter = torch.gather(pooled_parameters_channel, 1, segment.view(self.opt.batch_size, -1))
            unpooled_parameters.append(unpooled_parameter.view(self.opt.batch_size, 1, self.opt.height, self.opt.width))
        unpooled_parameters = torch.cat(unpooled_parameters, dim=1)

        # recover depth from plane parameters
        K_inv_dot_xy1 = torch.matmul(inputs[("inv_K", source_scale)][:, :3, :3],
                                     self.backproject_depth[source_scale].pix_coords)
        depth = 1. / (torch.sum(K_inv_dot_xy1 * unpooled_parameters.view(self.opt.batch_size, 3, -1), dim=1) + 1e-6)

        # clip depth range
        depth = torch.clamp(depth, self.opt.min_depth, self.opt.max_depth)
        depth = depth.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)
        outputs[("planar_depth", frame_id, scale)] = depth

    def compute_smooth_norm(self, inputs, outputs):
        """
        in:
            cam_points        b*4*H*W         tensor
            vps               b*6*3           tensor
        out:
            pred_norm         b*3*H*W         tensor
            aligned_norm       b*3*H*W         tensor
        """
        for scale in range(self.num_scales):
            cam_points = outputs[("cam_points", scale)]
            vps = inputs[("vps", 0, 0)]
            pred_norm = depth2norm(cam_points, self.opt.height, self.opt.width, self.opt.d2n_nei)
            outputs[("pred_norm", 0, scale)] = pred_norm

            mmap, mmap_mask, mmap_mask_thresh = compute_mmap(self.opt.batch_size, pred_norm, vps, self.opt.height, self.opt.width, self.epoch, self.opt.d2n_nei)
            if self.mmap_thresh_save_flag:
              with open(os.path.join(self.log_path, 'eval_res_for_each_epoch.txt'), 'a') as f:
                  f.write('\nepoch {} time {}'.format(self.epoch, time.asctime( time.localtime(time.time()) )))
                  f.write('\nmmap_mask_thresh {}'.format(mmap_mask_thresh))
                  self.mmap_thresh_save_flag = False
                
            aligned_norm = align_smooth_norm(self.opt.batch_size, mmap, vps, self.opt.height, self.opt.width)
            outputs[("aligned_norm", 0, scale)] = aligned_norm
            outputs[("mmap", 0, scale)] = mmap
            outputs[("mmap_mask", 0, scale)] = mmap_mask

    def compute_seg(self, inputs, outputs):
        """
        inputs:
            cam_points             b, 4, H*W
            aligned_norm        b, 3, H, W
            rgb                               b, 3, H, W
        outputs:
            seg                b, 1, H, W
        """
        nei = self.opt.d2n_nei
        for scale in range(self.num_scales):
            cam_points = outputs[("cam_points", scale)]
            aligned_norm = outputs[("aligned_norm", 0, scale)]
            rgb = inputs[("color_aug", 0, 0)]
            # calculate D using aligned norm
            D = compute_D(cam_points, aligned_norm)
            D = D.reshape(self.opt.batch_size, self.opt.height, self.opt.width)
            # move valid border from depth2norm neighborhood
            rgb = rgb[:, :, 2*nei:, :-2*nei]
            D = D[:, 2*nei:, :-2*nei]
            aligned_norm = aligned_norm[:, :, 2*nei:, :-2*nei]
            # comute cost

            rgb_down = self.pdist(rgb[:, :, 1:], rgb[:, :, :-1])
            rgb_right = self.pdist(rgb[:, :, :, 1:], rgb[:, :, :, :-1])

            rgb_down = torch.stack([normalize(rgb_down[i]) for i in range(self.opt.batch_size)])
            rgb_right = torch.stack([normalize(rgb_right[i]) for i in range(self.opt.batch_size)])

            D_down = abs(D[:, 1:] - D[:, :-1])
            D_right = abs(D[:, :, 1:] - D[:, :, :-1])
            norm_down = self.pdist(aligned_norm[:, :, 1:], aligned_norm[:, :, :-1])
            norm_right = self.pdist(aligned_norm[:, :, :, 1:], aligned_norm[:, :, :, :-1])

            D_down = torch.stack([normalize(D_down[i]) for i in range(self.opt.batch_size)])
            norm_down = torch.stack([normalize(norm_down[i]) for i in range(self.opt.batch_size)])

            D_right = torch.stack([normalize(D_right[i]) for i in range(self.opt.batch_size)])
            norm_right = torch.stack([normalize(norm_right[i]) for i in range(self.opt.batch_size)])

            normD_down = D_down + norm_down
            normD_right = D_right + norm_right

            normD_down = torch.stack([normalize(normD_down[i]) for i in range(self.opt.batch_size)])
            normD_right = torch.stack([normalize(normD_right[i]) for i in range(self.opt.batch_size)])

            # get max from (rgb, normD)
            cost_down = torch.stack([rgb_down, normD_down])
            cost_right = torch.stack([rgb_right, normD_right])
            cost_down, _ = torch.max(cost_down, 0)
            cost_right, _ = torch.max(cost_right, 0)
            # get dissimilarity map visualization
            dst = cost_down[:,  :,  : -1] + cost_right[ :, :-1, :]
            outputs[('seg_dst', 0, scale)] = dst
            # felz_seg
            cost_down_np = cost_down.detach().cpu().numpy()
            cost_right_np = cost_right.detach().cpu().numpy()
            segment = torch.stack([torch.from_numpy(felz_seg(normalize(cost_down_np[i]), normalize(cost_right_np[i]), 0, 0, self.opt.height-2*nei, self.opt.width-2*nei, scale =1,min_size=50)).cuda() for i in range(self.opt.batch_size)])
            # pad the edges that were previously trimmed
            segment += 1
            segment = F.pad(segment, (0,2*nei,2*nei,0), "constant", 0)
            outputs[("disp2seg", 0, scale)] = segment

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            sparse_reprojection_losses = []

            source_scale = 0

            disp = outputs[("disp", 0, scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            dso_target = outputs[("dso_color", 0, scale)]

            # dso loss
            for frame_id in self.opt.frame_ids_to_train[1:]:
                dso_pred = outputs[("dso_color", frame_id, scale)]
                sparse_reprojection_losses.append(self.compute_sparse_reprojection_loss(dso_pred, dso_target))

            if len(self.opt.frame_ids_to_train) == 5:
                dso_combined_1 = torch.cat((sparse_reprojection_losses[1], sparse_reprojection_losses[2]), dim=1)
                dso_combined_2 = torch.cat((sparse_reprojection_losses[0], sparse_reprojection_losses[3]), dim=1)

                dso_to_optimise_1, _ = torch.min(dso_combined_1, dim=1)
                dso_to_optimise_2, _ = torch.min(dso_combined_2, dim=1)
                dso_loss_1 = dso_to_optimise_1.mean() 
                dso_loss_2 = dso_to_optimise_2.mean()

                loss += dso_loss_1 + dso_loss_2
                losses["dso_loss_1/{}".format(scale)] = dso_loss_1
                losses["dso_loss_2/{}".format(scale)] = dso_loss_2
            else:
                dso_combined_1 = torch.cat(sparse_reprojection_losses, dim=1)
                dso_to_optimise_1, _ = torch.min(dso_combined_1, dim=1)
                dso_loss_1 = dso_to_optimise_1.mean()
                loss += dso_loss_1 
                losses["dso_loss_1/{}".format(scale)] = dso_loss_1
            
            # smooth loss
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            losses["smooth_loss/{}".format(scale)] = smooth_loss
            
            # planar loss
            if self.opt.using_disp2seg:
                loss_planar_reg = 0.0
                for frame_id in [0]:
                    pred_depth = outputs[("depth", frame_id, scale)]
                    planar_depth = outputs[("planar_depth", frame_id, scale)]
                    planar_mask = outputs[("planar_mask", frame_id, scale)] 
                    
                    assert torch.isnan(pred_depth).sum()==0, print(pred_depth)
                    
                    if torch.any(torch.isnan(planar_depth)):
                      print('warning! nan in planar_depth!')
                      planar_depth = torch.where(torch.isnan(planar_depth), torch.full_like(planar_depth, 0), planar_depth)
                      pred_depth = torch.where(torch.isnan(planar_depth), torch.full_like(pred_depth, 0), pred_depth)
                      
                    outputs[("planar_loss", frame_id, scale)] = torch.abs(pred_depth - planar_depth) * planar_mask
                    loss_planar_reg += torch.mean(outputs[("planar_loss", frame_id, scale)])
                loss += loss_planar_reg * self.opt.lambda_planar_reg
                losses["planar_reg_loss/{}".format(scale)] = loss_planar_reg
            
            # norm loss
            if self.opt.using_normloss:
                loss_norm_reg = 0.0
                for frame_id in [0]:
                    pred_norm = outputs[("pred_norm", frame_id, scale)]
                    aligned_norm = outputs[("aligned_norm", frame_id, scale)]
                    mmap_mask = outputs[("mmap_mask", frame_id, scale)]

                    if self.opt.using_disp2seg:
                        planar_mask = outputs[("planar_mask", frame_id, scale)] 
                        normloss_mask =  mmap_mask * planar_mask
                    else:
                        normloss_mask =  mmap_mask
                    
                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                    norm_loss_score = cos(pred_norm, aligned_norm)
                    
                    if torch.any(torch.isnan(norm_loss_score)):
                      print('warning! nan is norm loss compute! set nan = 1')
                      norm_loss_score = torch.where(torch.isnan(norm_loss_score), torch.full_like(norm_loss_score, 1), norm_loss_score)
                      
                    outputs[("norm_loss", frame_id, scale)] = (1 - norm_loss_score).unsqueeze(1) * normloss_mask
                    loss_norm_reg += torch.mean(outputs[("norm_loss", frame_id, scale)])
                loss += loss_norm_reg * self.opt.lambda_norm_reg
                losses["norm_reg_loss/{}".format(scale)] = loss_norm_reg

            total_loss += loss

            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_sparse_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        l1_loss = l1_loss.mean(3, True)
        ssim_loss = self.ssim_sparse(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        errors = []
        with torch.no_grad():
            for ind, (data, gt_depth, K, K_inv) in enumerate(tqdm(self.val_dataloader)):
                input_color = data.cuda()

                output = self.models["depth"](self.models["encoder"](input_color))

                pred_disp = F.interpolate(output[("disp", 0)], (gt_depth.shape[2], gt_depth.shape[3]))
                pred_disp, _ = disp_to_depth(pred_disp, self.opt.min_depth, self.opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                pred_depth = 1 / pred_disp
                pred_depth = pred_depth[0]

                gt_depth = gt_depth.data.numpy()[0, 0]

                mask = gt_depth > 0
                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]

                ratio = np.median(gt_depth) / np.median(pred_depth)
                pred_depth *= ratio

                pred_depth[pred_depth < self.opt.min_depth] = self.opt.min_depth
                pred_depth[pred_depth > self.opt.max_depth] = self.opt.max_depth

                errors.append(compute_errors(gt_depth, pred_depth))

        mean_errors = np.array(errors).mean(0)
 
        print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "lg10", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
        
        # write eval result to txt
        with open(os.path.join(self.log_path, 'eval_res_for_each_epoch.txt'), 'a') as f:
            f.write('\ntime {}'.format(time.asctime( time.localtime(time.time()) )))
            f.write("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "lg10", "a1", "a2", "a3")+'\n')
            f.write(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()))
            f.write('\n--------------------------------------------------------')

        # write to tensorboard
        writer = self.writers["val"]
        for l, v in zip(["abs_rel", "sq_rel", "rmse", "rmse_log", "lg10", "a1", "a2", "a3"],
                        mean_errors.tolist()):
            if l in ["abs_rel", "sq_rel", "rmse", "rmse_log", "lg10"]:
                writer.add_scalar("error/{}".format(l), v, self.step)
            else:
                writer.add_scalar("acc/{}".format(l), v, self.step)

        self.set_train()

    def log_time(self, batch_idx, duration, losses):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, losses["loss"].cpu().data,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

        writer = self.writers["train"]
        for l, v in losses.items():
            writer.add_scalar("loss/{}".format(l), v, self.step)

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("loss/{}".format(l), v, self.step)

        for j in range(min(1, self.opt.batch_size)):  # write a maxmimum of four images
            '''
            writer.add_image(
                "svo_{}/{}".format(0, j), 
                 inputs['svo_map'][j].unsqueeze(0).data, self.step)
            writer.add_image(
                "svo_noise_{}/{}".format(0, j), 
                 inputs['svo_map_noise'][j].unsqueeze(0).data, self.step)
            '''
            for s in [0]:
                if  self.opt.using_disp2seg: 
                    writer.add_image(                                                            
                        "planar_depth_{}/{}".format(s, j),
                        normalize_image(torch.clamp(outputs[("planar_depth", 0, s)][j], outputs[("depth", 0, s)][j].min().item(), outputs[("depth", 0, s)][j].max().item())), self.step)
                    writer.add_image(
                        "planar_mask_{}/{}".format(s, j),
                        outputs[("planar_mask", 0, s)][j], self.step)
                
                for frame_id in [0]:

                    if self.opt.using_disp2seg: 
                        writer.add_image(
                            "planar_loss_{}/{}".format(s, j),
                            normalize_image(outputs[("planar_loss", 0, s)][j]), self.step)
                        if self.opt.using_disp2seg:
                            writer.add_image(
                                "segment_{}/{}".format(s, j),
                                normalize_image(outputs[("disp2seg", 0, s)][j].unsqueeze(0)), self.step)

                    if self.opt.using_normloss or self.opt.using_disp2seg:
                        writer.add_image(
                            "pred_norm_{}_{}/{}".format(frame_id, s, j),
                            normalize_image(outputs[("pred_norm", frame_id, s)][j]), self.step)
                        writer.add_image(
                            "smooth_norm_{}_{}/{}".format(frame_id, s, j),
                            normalize_image(outputs[("aligned_norm", frame_id, s)][j]), self.step)
                            
                    if self.opt.using_normloss:
                        writer.add_image(
                            "norm_loss_{}_{}/{}".format(frame_id, s, j),
                            normalize_image(outputs[("norm_loss", frame_id, s)][j]), self.step)

                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    writer.add_image(
                        "depth_{}_{}/{}".format(frame_id, s, j),
                        normalize_image(outputs[("depth", frame_id, s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt_{}.json'.format(self.opt.start_epoch)), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

