from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import LinearNDInterpolator



cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")



# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


# plot_overview(current_kitti, original_attention_mask, casted_attention_mask, pred_disps[i], gt, )




def plot_overview(original_kitti, original_attention_mask, casted_attention_mask, pred_disps, gt_depth, mask, i, original_mask):
    fig, axis = plt.subplots(3, 2, figsize=(18, 9.5))

    path = f'hidde_depth_imgs/{i}'
    if not os.path.exists(path):
        os.mkdir(path)

    current_kitti = original_kitti
    current_kitti = np.swapaxes(current_kitti, 0, 1)
    current_kitti = np.swapaxes(current_kitti, 1, 2)

    np.save(os.path.join(path, 'kitti.npy'), current_kitti)

    font_nr = 15

    # axis[0].title.set_text('Original Kitti Image')
    axis[0, 0].set_title('Kitti image', fontdict={'fontsize': font_nr})
    axis[0, 0].axis('off')
    axis[0, 0].imshow(current_kitti)

    # create the heatmap
    # axis[1].title.set_text('Weight mask before pre processing')
    # axis[1].set_title('Attention mask', fontsize=font_nr)
    axis[0, 1].set_title('Weight mask before pre processing', fontdict={'fontsize': font_nr})
    axis[0, 1].axis('off')
    sns.heatmap(original_attention_mask, ax=axis[0, 1], vmin=1, vmax=1.2, cmap='Greens', center=1, cbar=False)
    np.save(os.path.join(path, 'original_attention_mask.npy'), original_attention_mask)


    # axis[2].set_title('Weight mask after pre processing', fontdict={'fontsize': 20, 'fontweight': 'bold'})
    axis[1,1].set_title('Weight mask after pre processing', fontdict={'fontsize': font_nr})
    # axis[2].title.set_text('Weight mask before pre processing')
    axis[1,1].axis('off')
    sns.heatmap(casted_attention_mask, ax=axis[1,1], vmin=0.8, vmax=1.2, cmap='Greens', center=1, cbar=False)
    np.save(os.path.join(path, 'casted_attention_mask.npy'), casted_attention_mask)


    # axis[3].set_title('Depth prediction', fontdict={'fontsize': 20})
    # # axis[3].title.set_text('Depth prediction')
    # axis[3].axis('off')
    # axis[3].imshow(pred_disps)

    depth_map = gt_depth / 256
    x, y = np.where(depth_map > 0)
    d = depth_map[depth_map != 0]

    xyd = np.stack((y, x, d)).T

    gt = lin_interp(depth_map.shape, xyd)

    axis[1,0].set_title('Depth label', fontdict={'fontsize': font_nr})
    # axis[4].title.set_text('LABEL')
    axis[1,0].axis('off')
    axis[1,0].imshow(gt, cmap='plasma')
    np.save(os.path.join(path, 'ground_truth.npy'), gt)


    axis[2,0].set_title('Depth label', fontdict={'fontsize': font_nr})
    # axis[5].title.set_text('Values left over')
    axis[2,0].axis('off')
    sns.heatmap(torch.tensor(original_mask).to(torch.float32), ax=axis[2,0], vmin=0.8, vmax=1.2, cmap='Greens', center=1, cbar=False)
    np.save(os.path.join(path, 'original_mask.npy'), original_mask)

    update_mask = torch.tensor(mask).to(torch.float32)

    axis[2,1].set_title('Depth values inside attention mask', fontdict={'fontsize': font_nr})
    # axis[6].title.set_text('Values left over with attention mask')
    axis[2,1].axis('off')
    sns.heatmap(update_mask, ax=axis[2,1], vmin=0.8, vmax=1.2, cmap='Greens', center=1, cbar=False)
    np.save(os.path.join(path, 'update_mask.npy'), update_mask)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.4)

    fig.savefig(f'{path}/{i}')
    plt.close('all')
    # breakpoint()



def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """



    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean( np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """

    if not os.path.exists('hidde_depth_imgs'):
        os.mkdir('hidde_depth_imgs')

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        # breakpoint()
        dataset = datasets.KITTIRAWDataset(opt.convolution_experiment, opt.top_k, opt.seed, opt.weight_mask_method, opt.weight_matrix_path, opt.attention_mask_loss, opt.edge_loss,  opt.data_path, opt.attention_path, opt.attention_threshold, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, img_ext='.png'if opt.png else '.jpg', is_train=False)

        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder#.cuda()
        encoder.eval()
        depth_decoder#.cuda()
        depth_decoder.eval()

        pred_disps = []
        original_attention_masks = []
        casted_attention_masks = []
        original_kitti = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():

            for i, data in enumerate(dataloader):

                print(i / len(dataloader))
                input_color = data[("color", 0, 0)]#.cuda()

                weight_masks = data['weight_matrix'].clone()

                # value < 1.05 to 0
                weight_masks[weight_masks <= 1.05] = 0

                # > 1 = 1
                weight_masks[weight_masks > 1] = 1

            # breakpoint()

                # FALSE
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                # get depth output from model
                output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                casted_attention_masks.append(weight_masks)

                original_masks = data['weight_matrix'].clone()
                original_attention_masks.append(original_masks)
                original_kitti.append(input_color.cpu())

                # # breakpoint()
                # if i ==1:
                #     break

        # AMOUNT IMG, 192, 640
        original_attention_masks = np.concatenate(original_attention_masks)
        pred_disps = np.concatenate(pred_disps)
        casted_attention_masks = np.concatenate(casted_attention_masks)
        original_kitti = np.concatenate(original_kitti)

        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
        # breakpoint()

        # for image in range(pred_disps.shape[0]):
        #
        #
        #
        #
        #     fig, axis = plt.subplots(5, 1, figsize=(16, 16))
        #
        #
        #
        #     depth_image = pred_disps[image]
        #
        #     current_kitti = original_kitti[image]
        #     current_kitti = np.swapaxes(current_kitti, 0, 1)
        #     current_kitti = np.swapaxes(current_kitti, 1, 2)
        #
        #
        #     axis[0].title.set_text('Original Kitti Image')
        #     axis[0].axis('off')
        #     axis[0].imshow(current_kitti)
        #
        #     # create the heatmap
        #     axis[1].title.set_text('Attention mask weight')
        #     axis[1].axis('off')
        #     sns.heatmap(attention_masks[image], ax=axis[1], vmin=1, vmax=1.2, cmap='Greens', center=1)
        #
        #
        #     axis[2].title.set_text('Depth image')
        #     axis[2].axis('off')
        #     axis[2].imshow(depth_image)
        #
        #
        #     depth_map = gt_depths[image]
        #
        #     depth_map = depth_map / 256
        #
        #     # breakpoint()
        #     x, y = np.where(depth_map > 0)
        #     d = depth_map[depth_map != 0]
        #
        #     xyd = np.stack((y, x, d)).T
        #
        #     gt = lin_interp(depth_map.shape, xyd)
        #
        #     axis[3].title.set_text('Input')
        #     axis[3].axis('off')
        #     axis[3].imshow(depth_map, cmap='plasma')
        #
        #     axis[4].title.set_text('Ground Truth')
        #     axis[4].axis('off')
        #     axis[4].imshow(gt, cmap='plasma')
        #
        #     fig.savefig('depth_evaluation/{}'.format(image))

    # FALSE
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    # FALSE
    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)
    # FALSE
    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    # EIGEN
    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")

    # shape = (697,) , load in the gt
    # ~ (375, 1242)
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    print("-> Evaluating")

    # FALSE
    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR

    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    # breakpoint()
    # for i, depth_map in enumerate(gt_depths):
    #     depth_map = depth_map[1] / 256
    #     x, y = np.where(depth_map > 0)
    #     d = depth_map[depth_map != 0]
    #
    #     xyd = np.stack((y, x, d)).T
    #
    #     gt = lin_interp(depth_map.shape, xyd)
    #
    #     fig, axis = plt.subplots(3, 1, figsize=(16, 16))
    #     axis[0].title.set_text('Original Kitti Image')
    #     axis[0].axis('off')
    #     axis[0].imshow(current_kitti)
    #
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(depth_map, cmap='plasma')
    #     plt.title("Input", fontsize=22)
    #     plt.show()
    #
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(gt)
    #     plt.title("Ground Truth", fontsize=22)
    #     plt.show()
    #
    #     fig.savefig('depth_evaluation/{}'.format(i))

    # loop over the predicted depth images
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]

        original_attention_mask = original_attention_masks[i]
        casted_attention_mask = casted_attention_masks[i]

        # resize back to depth label because during training kitti images were downsized
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        original_attention_mask = cv2.resize(original_attention_mask, (gt_width, gt_height))
        casted_attention_mask = cv2.resize(casted_attention_mask, (gt_width, gt_height))

        pred_depth = 1 / pred_disp


        # EIGEN
        if opt.eval_split == "eigen":

            #gt depth = 375, 1242

            # gt depth > 0.001 and gt depth < 80. # so filter out the pixels where the label is failing.
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)

            # select rows 153 t/m 371, colums = 44 t/m 1197 because the top rows are not captures by the label
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1

            # so if the label values are > 0 and < 80 and if it falls inside the row, col criterea
            mask = np.logical_and(mask, crop_mask)


        else:
            mask = gt_depth > 0


        original_mask = mask.copy()

        if opt.labels_inside_mask:

            # keep only the label values inside the attention mask
            mask = np.logical_and(mask, casted_attention_mask)

        # plot_overview(original_kitti[i], original_attention_mask, casted_attention_mask, pred_disps[i], gt_depth, mask,
        #               i, original_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor

        # FALSE DUS WORDT UITGEVOERD
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        # breakpoint()
        if gt_depth.shape[0] > 0 and pred_depth.shape[0] > 0:
            errors.append(compute_errors(gt_depth, pred_depth))


    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    # print("OPTIONS", options.parse())
    # evaluate(options)
    evaluate(options.parse())
