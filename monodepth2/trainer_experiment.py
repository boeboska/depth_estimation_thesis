# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


from __future__ import absolute_import, division, print_function

import numpy as np
import time
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import collections
from torchvision import transforms
import copy
import cv2
import matplotlib as mpl
import matplotlib.cm as cm

torch.backends.cudnn.enabled = False
torch.cuda.empty_cache()

torch.autograd.set_detect_anomaly(True)

from attention_mask_loss import *
from attention_weight_mask import *
from self_attention_util import *
from edge_code import edge_detection_bob_hidde
# import edge_code as edge_code

import pickle


class Trainer:

    def __init__(self, options):

        self.opt = options
        print(self.opt.log_dir, self.opt.model_name)
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        torch.manual_seed(self.opt.seed)
        torch.cuda.manual_seed(self.opt.seed)
        torch.cuda.manual_seed_all(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")


        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.self_attention == True:
            self.models["encoder"] = networks.ResnetEncoderSelfAttention(
                self.opt.num_layers, self.opt.weights_init == "pretrained", top_k=self.opt.top_k)
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())


        else:
            # print("JAAJAJAJ")
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained", top_k = self.opt.top_k)
            self.models["encoder"].to(self.device)
            self.parameters_to_train += list(self.models["encoder"].parameters())

            print(sum(p.numel() for p in self.models["encoder"].parameters() if p.requires_grad))

        # print("ENCODEERTJE HIDDE", self.models["encoder"])
        # breakpoint()
        if self.opt.self_attention:
            self.models["depth"] = networks.DepthDecoderSelfAttention(
                self.models["encoder"].num_ch_enc, self.opt.self_attention, self.opt.scales)
        else:

            self.models["depth"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames, top_k = self.opt.top_k)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        # train_filenames = readlines(fpath.format("train_small"))
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        test_filesnames = readlines(fpath.format("update_test"))

        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.convolution_experiment, self.opt.top_k,
            self.opt.seed, self.opt.weight_mask_method, self.opt.weight_matrix_path, self.opt.attention_mask_loss, self.opt.edge_loss,
            self.opt.data_path, self.opt.attention_path, self.opt.attention_threshold, train_filenames, self.opt.height,
            self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            self.opt.convolution_experiment,
            self.opt.top_k,
            self.opt.seed,
            self.opt.weight_mask_method,
            self.opt.weight_matrix_path,
            self.opt.attention_mask_loss,
            self.opt.edge_loss,
            self.opt.data_path,
            self.opt.attention_path,
            self.opt.attention_threshold,
            val_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=False,
            img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.val_iter = iter(self.val_loader)


        test_dataset = self.dataset(
            self.opt.convolution_experiment,
            self.opt.top_k,
            self.opt.seed,
            self.opt.weight_mask_method,
            self.opt.weight_matrix_path,
            self.opt.attention_mask_loss,
            self.opt.edge_loss,
            self.opt.data_path,
            self.opt.attention_path,
            self.opt.attention_threshold,
            test_filesnames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,
            is_train=False,
            img_ext=img_ext)
        self.test_loader = DataLoader(
            test_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.test_iter = iter(self.test_loader)
        #
















        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

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

        self.val_losses = []

        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            message = self.run_epoch()
            if message == "stop":
                return "stop"
            # if (self.epoch + 1) % self.opt.save_frequency == 0:
            self.save_model(batch_idx = 9999)

    def early_stopping_check(self, batch_idx):

        diff_1 = (self.val_losses[-1] - self.val_losses[-2]) / self.val_losses[-2]
        diff_2 = (self.val_losses[-2] - self.val_losses[-3]) / self.val_losses[-3]
        diff_3 = (self.val_losses[-3] - self.val_losses[-4]) / self.val_losses[-4]
        # print(diff_1, diff_2, diff_3)

        if abs(diff_1) and abs(diff_2) and abs(diff_3) < self.opt.early_stop_percentage:
            self.save_model(batch_idx)
            return "stop"
        else:
            return "continue"



    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        # hist_dict = {}
        #
        # weight_size = np.arange(0, 16000, 1)
        # attention_sizes = np.arange(0, 1.01, 0.01)
        #
        # # 1 ... 16000
        # for weight_mask_size in weight_size:
        #
        #     hist_dict[weight_mask_size] = {}
        #
        #     # 0.05 ... 5
        #     for attention_size in attention_sizes:
        #         hist_dict[weight_mask_size][attention_size] = []

        hist_dict = None


        for batch_idx, inputs in enumerate(self.train_loader):

            # print("IDX ", batch_idx)




            # breakpoint()
            # if batch_idx % 50 == 0:
            #
            #     weight_folder = self.opt.load_weights_folder.split('monodepth_models/')[1].split('/')[0]
            #     epoch_nr =self.opt.load_weights_folder.split('monodepth_models/')[1].split('weights_')[1].split('_')[0]
            #     with open('validation_all/'  +  'hist_dict_attention_map' + 'exp_' + str(weight_folder) + '_' + str(batch_idx) + 'epoch_ ' + str(epoch_nr) + '.pkl', 'wb') as f:
            #         pickle.dump(hist_dict, f, pickle.HIGHEST_PROTOCOL)

            before_op_time = time.time()

            outputs, losses, hist_dict = self.process_batch(inputs, batch_idx, hist_dict)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                #
                self.log("train", inputs, outputs, losses)

                # print("ik ga valideren")
                self.val_losses.append(self.val(inputs, batch_idx, hist_dict))
                if len(self.val_losses) >= 4:
                    message = self.early_stopping_check(batch_idx)
                    if message == "stop":
                        return "stop"
                # print("vall loss", self.val_losses)

            self.step += 1



    def process_batch(self, inputs, batch_idx, hist_dict):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # breakpoint()

        # NO
        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])

        else:

            if self.opt.top_k > 0:
                # feed the three kitti images to the decoder
                # breakpoint()
                features = self.models["encoder"](inputs["color_aug", 0, 0], masks = inputs['top_k_masks'], batch_idx = batch_idx)

            else:
                # Otherwise, we only feed the image with frame_id 0 through the depth encoder
                # breakpoint()
                # print("OPTIONS SELF ATTENTION ", self.opt.self_attention)

                # print("ENDCODER WEER ", self.models["encoder"])
                if self.opt.self_attention:
                    # print("ENCODER SELF ATTENTION GA IK DOEN")

                    features, attention_maps, hist_dict = self.models["encoder"](inputs["color_aug", 0, 0], batch_idx = batch_idx, inputs = inputs, hist_dict = hist_dict)
                else:

                    features = self.models["encoder"](inputs["color_aug", 0, 0])

            if self.opt.self_attention:
                outputs = self.models["depth"](features, attention_maps)

                # if batch_idx % self.opt.save_plot_every == 0:
                #     save_self_attention_masks(inputs, outputs, batch_idx, self.epoch, self.opt.model_name, self.opt.log_dir)

            else:

                outputs = self.models["depth"](features)

        # FALSE
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # TRUE
        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features, batch_idx))

        self.generate_images_pred(inputs, outputs, batch_idx)

        if self.opt.self_attention == False:
            attention_maps = None
        losses = self.compute_losses(inputs, outputs, batch_idx, hist_dict, attention_maps)

        return outputs, losses, hist_dict

    def predict_poses(self, inputs, features, batch_idx):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        #TRUE
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}

             # True
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    # breakpoint()
                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)

                    # self.pose_depth_info[str(batch_idx) + 'axisangle'] = axisangle.reshape(6)
                    # self.pose_depth_info[str(batch_idx) + 'translation'] = translation.reshape(6)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # breakpoint()

                    # Invert the matrix if the frame id is negative
                    # select the first 3 axis angle and trans params
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs


    def update_dict(self, losses, loss_per_mask_size):

        empty_dict_losses = {
            'loss': None,
            'loss_within_attention_mask': None,
            'loss_within_attention_mask_dialation_1': None,
            'loss_within_attention_mask_dialation_3': None,
            'total_loss_without_edge': None
        }

        # if this amount of pixels is not yet seen
        if losses['amount_pixels_inside_mask'] not in loss_per_mask_size:

            # fill dict with the numbers
            for key in empty_dict_losses:
                # breakpoint()
                empty_dict_losses[key] = [losses[key].item()]

            # add to the main dict
            loss_per_mask_size[losses['amount_pixels_inside_mask']] = empty_dict_losses

        else:
            # get current dict
            curr_dict = loss_per_mask_size[losses['amount_pixels_inside_mask']]

            # add values to the current dict
            for key, value in curr_dict.items():
                current_list = curr_dict[key]
                current_list.append(losses[key].item())
                curr_dict[key] = current_list

            # add the new values to the total dict
            loss_per_mask_size[losses['amount_pixels_inside_mask']] = curr_dict

        return loss_per_mask_size

    def test_all(self, current_model):

        loss_per_mask_size = {}
        edge_loss_total = []

        self.set_eval()
        start = time.time()

        hist_dict = {}

        weight_size = np.arange(0, 16000, 1)
        attention_sizes = np.arange(0, 1.01, 0.01)

        # 1 ... 16000
        for weight_mask_size in weight_size:

            hist_dict[weight_mask_size] = {}

            # 0.05 ... 5
            for attention_size in attention_sizes:
                hist_dict[weight_mask_size][attention_size] = []




        # loop over all the validation images
        for batch_idx, inputs in enumerate(self.test_loader):


            if batch_idx % 250 == 0:

                print(batch_idx)
            with torch.no_grad():
                outputs, losses, hist_dict = self.process_batch(inputs, batch_idx, hist_dict)

                # breakpoint()
                # edge_loss_total.append(['total_loss_without_edge'])
                # loss_per_mask_size = self.update_dict(losses, loss_per_mask_size)


        # save the dictionary
        with open('validation_all/' + 'test_' + current_model + 'hist_dict' + '.pkl', 'wb') as f:
            pickle.dump(hist_dict, f, pickle.HIGHEST_PROTOCOL)


        return None

    def val_all(self, current_model):
        """Validate on the whole validation set"""

        # loss_per_mask_size = {}
        # validation_edge_loss = []

        # avg_dialation_size = {}
        # avg_dialation_size["additional_dialation_1"] = []
        # avg_dialation_size["additional_dialation_3"] = []
        # avg_dialation_size["cover_no_dial"] = []
        # avg_dialation_size["cover_dial_1"] = []
        # avg_dialation_size["cover_dial_3"] = []

        hist_dict = {}

        weight_size = np.arange(0, 16000, 1)
        attention_sizes = np.arange(0, 1.01, 0.01)

        # 1 ... 16000
        for weight_mask_size in weight_size:

            hist_dict[weight_mask_size] = {}

            # 0.05 ... 5
            for attention_size in attention_sizes:
                hist_dict[weight_mask_size][attention_size] = []

        self.set_eval()
        start = time.time()

        # loop over all the validation images
        for batch_idx, inputs in enumerate(self.val_loader):

            if batch_idx % 250 == 0:
                # with open('validation_all/' + current_model + str(batch_idx) +  'hist_dict' + '.pkl', 'wb') as f:
                #     pickle.dump(hist_dict, f, pickle.HIGHEST_PROTOCOL)
                print(batch_idx)
            with torch.no_grad():

                outputs, losses, hist_dict  = self.process_batch(inputs, batch_idx, hist_dict)

                # add edge loss for current image
                # validation_edge_loss.append(losses['loss'])
                # breakpoint()



                # loss_per_mask_size = self.update_dict(losses, loss_per_mask_size)

            # if batch_idx == 10:



        # save the dictionary
        with open('validation_all/' + current_model + 'hist_dict' + '.pkl', 'wb') as f:
            pickle.dump(hist_dict, f, pickle.HIGHEST_PROTOCOL)

        # save list
        # validation_edge_loss = np.array(validation_edge_loss)
        # np.save('validation_all/' + current_model, validation_edge_loss)

        return None

    def val(self, inputs, batch_idx, hist_dict):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            # breakpoint()
            outputs, losses, hist_dict = self.process_batch(inputs, batch_idx, hist_dict)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)

            # breakpoint()
            val_loss = losses['loss'].item()
            del inputs, outputs, losses

        self.set_train()
        return val_loss

    def plot_attention_masks(self, inputs, batch_idx):

        path = self.opt.log_dir + self.opt.model_name + "/" + "weight_matrix_img/"
        if not os.path.exists(path):
            os.makedirs(path)

        for b in range(self.opt.batch_size):
            original_img = inputs["color_aug", 0, 0][b]

            original_img = np.array(original_img.cpu().detach().numpy())

            original_img = np.swapaxes(original_img, 0, 1)
            original_img = np.swapaxes(original_img, 1, 2)

            weight_mask = inputs['weight_matrix'][b].cpu()

            fig, axis = plt.subplots(1, 2, figsize=(40, 5))

            # create the heatmap
            sns.heatmap(weight_mask, ax=axis[0], vmin=1, vmax=1.2, cmap='Greens', center=1)

            # put the original rgb kitti image in the subplot
            axis[1].imshow(original_img)
            fig.savefig('{}/epoch_{}_batch_idx_{}_batch_{}.png'.format(path, self.epoch, batch_idx, b))
            plt.close(fig)

    def generate_images_pred(self, inputs, outputs, batch_idx):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                    # self.pose_depth_info[str(batch_idx) + 'T'] = T[0]


                # breakpoint()

                # from the authors of https://arxiv.org/abs/1712.00175
                # breakpoint()

                # FALSE
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)


                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def prepare_attention_masks(self, inputs):

        attention_mask_weight = inputs['weight_matrix'].to(self.device).clone()

        # for calculating monodepth loss inside attention mask
        attention_masks_for_calculating_loss_inside_mask = attention_mask_weight.clone()

        original_attention_mask = attention_mask_weight.clone()

        # values <= 1.05 to 0. so only left with the small attention masks
        attention_masks_for_calculating_loss_inside_mask[attention_masks_for_calculating_loss_inside_mask <= self.opt.attention_mask_threshold] = 0

        # > 0 = 1
        attention_masks_for_calculating_loss_inside_mask[attention_masks_for_calculating_loss_inside_mask > 0] = 1

        amount_pixels_inside_mask = attention_masks_for_calculating_loss_inside_mask.sum().item()

        attention_masks_for_calculating_loss_inside_mask_with_dilation_1 = attention_masks_for_calculating_loss_inside_mask.clone()
        original_mask_before_dialation = attention_masks_for_calculating_loss_inside_mask.clone()
        attention_masks_for_calculating_loss_inside_mask_with_dilation_3 = attention_masks_for_calculating_loss_inside_mask.clone()

        kernel = np.ones((3, 3), np.uint8)

        for b in range(attention_masks_for_calculating_loss_inside_mask_with_dilation_1.shape[0]):
            attention_masks_for_calculating_loss_inside_mask_with_dilation_1[b] = torch.tensor(cv2.dilate(attention_masks_for_calculating_loss_inside_mask_with_dilation_1[b].cpu().detach().numpy(), kernel, iterations=1))
            attention_masks_for_calculating_loss_inside_mask_with_dilation_1[b] = attention_masks_for_calculating_loss_inside_mask_with_dilation_1[b] - original_mask_before_dialation[b]

            attention_masks_for_calculating_loss_inside_mask_with_dilation_3[b] = torch.tensor(cv2.dilate(attention_masks_for_calculating_loss_inside_mask_with_dilation_3[b].cpu().detach().numpy(), kernel, iterations=3))
            attention_masks_for_calculating_loss_inside_mask_with_dilation_3[b] = attention_masks_for_calculating_loss_inside_mask_with_dilation_3[b] - original_mask_before_dialation[b]


        # alle pixel waarden tot aan 1.03 map maar 1
        attention_mask_weight[
            attention_mask_weight <= self.opt.attention_mask_threshold] = self.opt.reduce_attention_weight

        # geef extra weight mee aan pixels die nog over zijn gebleven
        attention_mask_weight[attention_mask_weight > 1] = attention_mask_weight[
                                                               attention_mask_weight > 1] * self.opt.attention_weight


        return attention_mask_weight, original_attention_mask, attention_masks_for_calculating_loss_inside_mask, attention_masks_for_calculating_loss_inside_mask_with_dilation_1, attention_masks_for_calculating_loss_inside_mask_with_dilation_3, amount_pixels_inside_mask


    def calculate_self_attention_size(self, hist_dict, attention_maps, amount_pixels_inside_mask, batch_idx):

        all_pixels = 512 * 24 * 80
        count = 0
        for i in range(len(list(hist_dict.keys()))):

            if amount_pixels_inside_mask > max(hist_dict):
                amount_pixels_inside_mask = max(hist_dict)

            attention_map = attention_maps.squeeze().cpu().clone()


            # normalized 0-1
            attention_map -= attention_map.min(1, keepdim=True)[0]
            attention_map /= attention_map.max(1, keepdim=True)[0]


            # nan to 0
            attention_map[attention_map != attention_map] = 0

            # path = f'attention_map_check/'
            # # path = self.log_dir + self.model_name + "/" + "vis_query/"
            # if not os.path.exists(path):
            #     os.makedirs(path)
            #
            # fig, axis = plt.subplots(11, 2, figsize=(30, 40))
            # for x in range(10):
            #     # for y in range(2):
            #
            #     rand_nr = randrange(attention_map.shape[0])
            #
            #     axis[x+1, 0].set_title(f'attention map')
            #     axis[x+1, 0].axis('off')
            #     # axis[x+1, 0].imshow(attention_map_eerst[rand_nr].cpu().detach().numpy())
            #     sns.heatmap(attention_map_eerst[rand_nr].cpu().detach().numpy(), ax=axis[x+1, 0], vmin=0, vmax=torch.max(attention_map_eerst[rand_nr]).item(), cmap='Greens', cbar=False)
            #
            #
            #     axis[x+1, 1].set_title(f'normalized map')
            #     axis[x+1, 1].axis('off')
            #     axis[x + 1, 1].imshow(attention_map[rand_nr].cpu().detach().numpy())
            #     sns.heatmap(attention_map[rand_nr].cpu().detach().numpy(), ax=axis[x+1, 1], vmin=0, vmax=1, cmap='Greens', cbar=False)
            #
            #
            #
            # fig.savefig(f'{path}_{batch_idx}_{rand_nr}.png')
            # plt.close(fig)

            # 0.05 ... 1
            list_of_keys = list(hist_dict[amount_pixels_inside_mask].keys()).copy()

            # make sure you also get items > 4
            if list_of_keys[i + 1] == list_of_keys[-1]:
                list_of_keys[i + 1] = np.inf
            # breakpoint()
            attention_map = (attention_map >= list_of_keys[i]) & (attention_map < list_of_keys[i + 1])

            count += attention_map.sum().item()

            curr = torch.div(attention_map.sum(dim=-1).sum(1).float(),
                             (attention_maps.shape[2] * attention_maps.shape[3]))

            # breakpoint()

            current_list = hist_dict[amount_pixels_inside_mask][list_of_keys[i]]
            current_list.append( torch.mean(curr).item() )

            hist_dict[amount_pixels_inside_mask][list_of_keys[i]] = current_list


            # now you have had all the data
            if list_of_keys[i + 1] == np.inf:
                break
        # breakpoint()

        # check if all values of the tensor are fallen inside the thresholds
        assert count == all_pixels, f"difference if {all_pixels - count}"

        return hist_dict


    def compute_losses(self, inputs, outputs, batch_idx, hist_dict, attention_maps):
        """Compute the reprojection and smoothness losses for a minibatch
        """

        # because once in the 250 steps you want to plot an edge loss
        # if batch_idx % self.opt.save_plot_every == 0 and self.opt.edge_loss:
        #     original_masks = torch.clone(inputs['attention'])
        # else:
        #     original_masks = None

        # plot during training the original kitti image with the attention mask weight

        losses = {}
        total_loss = 0
        total_loss_without_mask = 0
        total_loss_without_edge = 0

        total_loss_within_mask = 0
        total_loss_within_mask_dialation_1 = 0
        total_loss_within_mask_dialation_3 = 0

        total_edge_loss = 0
        total_attention_weight_loss = 0

        _, original_attention_masks, loss_inside_mask_tensor, loss_inside_mask_tensor_dialation_1, loss_inside_mask_tensor_dialation_3, amount_pixels_inside_mask= self.prepare_attention_masks(inputs)


        # first determine if you need to calculate the attention mask loss. Then you only have to do this once.
        # this is indepenedned of the scale or frame id.
        # batch size , 192, 640
        if self.opt.attention_mask_loss == True:

            attention_mask_weight, original_attention_masks, loss_inside_mask_tensor, loss_inside_mask_tensor_dialation_1, loss_inside_mask_tensor_dialation_3, amount_pixels_inside_mask = self.prepare_attention_masks(inputs)

            # hist_dict = self.calculate_self_attention_size(hist_dict, attention_maps, amount_pixels_inside_mask, batch_idx)
        else:
            attention_mask_weight = torch.ones(size=(self.opt.batch_size, 1, self.opt.height, self.opt.width)).to(
                self.device)


        # first determine if you need to calculate the edge loss. Then you only have to do this once.
        # so you will use the same not overlapping attention_masks for every edge loss scale
        # if self.opt.edge_loss == True:
            # first determine which masks over overlapping.
            # because you don't want overlapping masks for edge detection because you might find the same edge multiple times
            # not_overlapping_attention_masks, index_numbers_not_overlapping = overlapping_masks_edge_detection(self, inputs, batch_idx, original_masks)

        for scale in self.opt.scales:
            loss = 0
            loss_without_edge = 0
            loss_without_mask = 0
            loss_within_mask = 0
            loss_within_mask_dialation_1 = 0
            loss_within_mask_dialation_3 = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            # if scale == 0:
                # breakpoint()
                # self.pose_depth_info[str(batch_idx) + 'disp'] = [round(torch.min(disp).item(), 2), round(torch.max(disp).item(), 2), round(torch.mean(disp).item(), 2), round(torch.std(disp).item(), 2)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:

                pred = outputs[("color", frame_id, scale)]

                # print("PRED TARGET", pred.shape, target.shape)
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            # if scale == 0 and batch_idx % self.opt.save_plot_every == 0:
            #     plot_tensor_begin_training(self, inputs, outputs, batch_idx, scale, reprojection_losses)

            # skip first epoch because depth image is not converged and edges are noise now
            if self.opt.edge_loss == True and scale == 0 and self.epoch > 0:
            # if self.opt.edge_loss == True and scale == 0:

                edge_loss = edge_detection_bob_hidde(scale, outputs, inputs, batch_idx, self.device, self.opt.height, self.opt.width, self.opt.log_dir, self.opt.model_name, self.opt.edge_detection_threshold, self.opt.save_plot_every, self.opt.batch_size).to(self.device)

                loss += self.opt.edge_weight * edge_loss * self.num_scales

                total_edge_loss += self.opt.edge_weight * edge_loss * self.num_scales

                # add for writing to tensorboard
                losses["edge_loss/{}".format(scale)] = self.opt.edge_weight * edge_loss

            # this statement is performed
            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:

                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            # FALSE
            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing m ask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            # FALSE
            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            # this one is performed
            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).to(self.device) * 0.00001

                # if self.opt.no_cuda cpu() else cuda()

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            # FALSE
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                # breakpoint()
                to_optimise, idxs = torch.min(combined, dim=1)





            loss_without_mask += to_optimise.mean()
            loss_without_edge += to_optimise.mean()

            temp = to_optimise * loss_inside_mask_tensor
            loss_within_mask += torch.mean( temp [temp > 0] )


            temp = to_optimise * torch.tensor(loss_inside_mask_tensor_dialation_1).to(self.device)
            loss_within_mask_dialation_1 += torch.mean( temp [temp > 0] )


            temp = to_optimise * torch.tensor(loss_inside_mask_tensor_dialation_3).to(self.device)
            loss_within_mask_dialation_3 += torch.mean( temp [temp > 0] )

            # if amount_pixels_inside_mask > 12000 and scale == 0 or amount_pixels_inside_mask < 750 and scale == 0:
            # #
            #     # print(amount_pixels_inside_mask)
            #
            #     path = self.opt.log_dir + self.opt.model_name + "/" + "big_img_visualization/"
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            #
            #     path = f'{path}/{batch_idx}'
            #     if not os.path.exists(path):
            #         os.makedirs(path)
            # #
            #     fig, axis = plt.subplots(2, 1, figsize=(30, 20))
            #
            #     original_img = inputs["color_aug", 0, 0]
            #
            #     original_img = np.array(original_img[0].squeeze().cpu().detach().permute(1, 2, 0).numpy())
            #
            #     # axis[0, 0].title.set_text('Kitti image')
            #     axis[0].set_title('Kitti image', fontdict={'fontsize': 20, 'fontweight': 'bold'})
            #     axis[0].axis('off')
            #     axis[0].imshow(original_img)
            #     np.save(os.path.join(path, 'monodepth_loss.npy'), original_img)
            #
            #     axis[1].set_title(f'Attention mask: {amount_pixels_inside_mask}', fontdict={'fontsize': 20, 'fontweight': 'bold'})
            #     axis[1].axis('off')
            #     sns.heatmap(attention_mask_weight.cpu().squeeze(0).detach(), ax=axis[1], vmin=1, vmax=1.2, cmap='Greens',
            #                 center=1)
            #     np.save(os.path.join(path, 'attention_mask.npy'), loss_inside_mask_tensor[0].detach().cpu().numpy())
            # #
            #     fig.savefig(f'{path}/batch_idx_{batch_idx, scale, amount_pixels_inside_mask}.png')
            #     plt.close(fig)
            #









            if self.opt.attention_mask_loss:
                """ if reprojection loss is higher then the identity loss then this is due to no camera motion or moving objects at same speed.
                for these pixels we use the identity loss such that the loss is lower and no noise to the model
                so you don't want to mutliply the pixels which use the identity loss because then you upsize the identity loss for these pixels
                IDX 2,3 = loss reprojection < loss identity. This is fine it means that the estimation via de models is better then without the models
                IDX 0,1 = loss identity < loss reprojection which means that the estimation without models is better then with the models. 
                for these pixels skip the weight mask, so set to 1 such that loss after multiplication doens't change
                """


                if batch_idx % self.opt.save_plot_every == 0 and self.opt.attention_mask_loss and batch_idx != 0 and scale ==0:
                    plot_loss_tensor(self, inputs, to_optimise, original_attention_masks, batch_idx, scale, idxs, identity_reprojection_loss,
                                     loss_inside_mask_tensor, loss_inside_mask_tensor_dialation_1, loss_inside_mask_tensor_dialation_3)


                total_attention_weight_loss += (to_optimise * attention_mask_weight).mean() - to_optimise.mean()


                to_optimise = to_optimise * attention_mask_weight


        # check which pixel is the minimum and check if this is greater then the identity reprojection for u mask
        # This one is performed. create mask ims for tensorboard
            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss_without_mask += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            loss_without_edge += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)


            total_loss += loss
            total_loss_without_mask += loss_without_mask
            total_loss_without_edge += loss_without_edge

            total_loss_within_mask += loss_within_mask
            total_loss_within_mask_dialation_1 += loss_within_mask_dialation_1
            total_loss_within_mask_dialation_3 += loss_within_mask_dialation_3


        total_edge_loss /= self.num_scales
        total_loss_without_edge /= self.num_scales
        total_loss_within_mask /= self.num_scales
        total_loss_within_mask_dialation_1 /= self.num_scales
        total_loss_within_mask_dialation_3 /= self.num_scales
        total_loss /= self.num_scales
        total_loss_without_mask /= self.num_scales
        total_attention_weight_loss /= self.num_scales



        # dial_1 = ((loss_inside_mask_tensor_dialation_1 != 0).sum().item() )
        # dial_3 = ((loss_inside_mask_tensor_dialation_3 != 0).sum().item() )
        # no_dial = ((loss_inside_mask_tensor != 0).sum().item() )
        #
        # if no_dial > 0:
        #
        #     additional_dialation_1 =  (dial_1 - no_dial) / no_dial
        #     additional_dialation_3 = (dial_3 - no_dial) / no_dial
        #
        #     losses["additional_dialation_1"] = additional_dialation_1
        #     losses["additional_dialation_3"] = additional_dialation_3
        #
        #     losses["cover_no_dial"] = no_dial / (self.opt.width * self.opt.height)
        #     losses["cover_dial_1"] = dial_1 / (self.opt.width * self.opt.height)
        #     losses["cover_dial_3"] = dial_3 / (self.opt.width * self.opt.height)


        losses["total_loss_without_edge"] = total_loss_without_edge
        losses["amount_pixels_inside_mask"] = amount_pixels_inside_mask
        losses["loss_within_attention_mask"] = total_loss_within_mask
        losses["loss_within_attention_mask_dialation_1"] = total_loss_within_mask_dialation_1
        losses["loss_within_attention_mask_dialation_3"] = total_loss_within_mask_dialation_3

        losses["loss_without_attention_weight"] = total_loss_without_mask
        losses["loss"] = total_loss
        losses["total_additional_weight_loss"] = total_attention_weight_loss
        losses["edge_loss_total/{}"] = total_edge_loss


        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training
        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

        current_log = [item.replace('\n', '') for item in open('output_during_training.txt').readlines()]
        current_log.append(print_string)

        with open('output_during_training.txt', 'w') as file:
            file.write('\n'.join(current_log))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, batch_idx):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}_batch_idx{}".format(self.epoch, batch_idx))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

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

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")