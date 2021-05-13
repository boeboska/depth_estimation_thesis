import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import seaborn as sns
import cv2
import torch.nn.functional as F


def edge_detection_bob_hidde(scale, outputs, inputs, batch_idx, self):

    path = self.opt.log_dir + self.opt.model_name + "/" + "edge_loss_img/"
    if not os.path.exists(path):
        os.makedirs(path)

    print(inputs['attention'].shape)
    # breakpoint()
    attention_masks = inputs['attention'].to(self.device)

    attention_masks[attention_masks >= 0.8] = 1
    attention_masks[attention_masks < 0.8] = 0

    attention_masks_plot = attention_masks
    # breakpoint()

    attention_masks = torch.sum(attention_masks, dim = 1)

    attention_masks[attention_masks >= 1] = 1

    attention_masks_plot = attention_masks

    disp = outputs[("disp", scale)]
    # upsample to kitti size for correct multiplication with attention_mask
    disp = F.interpolate(
        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)



    depth_mask = attention_masks.squeeze(0) * disp.squeeze(0).squeeze(0)

    depth_mask_plot = depth_mask

    depth_mask = depth_mask * 255

    # cast to int for correct canny input
    depth_mask = np.uint8(depth_mask.cpu().detach().numpy())

    edges_disp = cv2.Canny(depth_mask, self.opt.edge_detection_threshold * 255, self.opt.edge_detection_threshold * 255,
                           apertureSize=3, L2gradient=False)

    fig, ax = plt.subplots(5, 1, figsize=(12, 12))

    original_img = inputs["color_aug", 0, 0]

    original_img = np.array(original_img.squeeze().cpu().detach().permute(1, 2, 0).numpy())

    # 2, 192, 640, 3
    ax[0].imshow(original_img)
    ax[0].title.set_text('Original image')
    ax[0].axis('off')

    ax[1].imshow(disp.squeeze(0).squeeze(0).cpu().detach().numpy())
    ax[1].title.set_text('depth imagw')
    ax[1].axis('off')

    ax[2].imshow(attention_masks_plot.squeeze(0).cpu().detach().numpy())
    ax[2].title.set_text('attention masks')
    ax[2].axis('off')

    disp_min = np.uint8(disp.min().cpu().detach().numpy() * 255)
    disp_max = np.uint8(disp.max().cpu().detach().numpy() * 255)

    ax[3].imshow(depth_mask_plot.cpu().detach().numpy(), vmin=disp_min, vmax=disp_max)
    ax[3].title.set_text('depth mask masks')
    ax[3].axis('off')


    ax[4].imshow(edges_disp / 255)
    ax[4].title.set_text('edges')
    ax[4].axis('off')
    #

    fig.savefig(
        '{}/epoch_{}_batchIDX_{}_threshold_{}.png'.format(path, self.epoch, batch_idx, self.opt.edge_detection_threshold))


    return None

def edge_detection_loss(self, scale, outputs, inputs, attention_mask, batch_idx, index_nrs_not_overlapping,
                        original_attention):
    """"
    scale: this selects the correct depth image
    outputs: for selecting the depth image
    inputs: for selecting the original kitti image
    attention_mask: these are the not overlapping attention masks
    batch_idx: iteration number during training. This is needed because every 250 steps, save the edge image
    index_nrs_not_overlapping: these are the index numbers of the not overlapping masks, needed for indexing
    original_attention: needed for plotting every 250 steps
    """

    edge_loss = []

    # set this on if you want to double check if the tensors are not overlapping.
    # self.additional_not_overlapping_check(attention_mask)

    attention_mask = torch.clone(attention_mask).to(self.device)

    # set all the negative values to zero. the values were negative to make sure no overlapping values were found
    # curing the attention_mask_weight function
    attention_mask[attention_mask <= -1] = 0

    # breakpoint()
    disp = outputs[("disp", scale)].to(self.device)

    # upsample to kitti size for correct multiplication with attention_mask
    disp = F.interpolate(
        disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False).to(self.device)

    # sum all the attention mask together in 1 attention tensor for faster computation. This is possible because there are not overlalping maps # attention_mask [2, 1, 192, 640]
    attention_mask = attention_mask.sum(1).unsqueeze(1).to(self.device)

    assert attention_mask.max() == 1, "sum of attention masks is greater than one .. probably overlapping masks"

    # create only the depth pixels that lie inside the attention mask
    depth_mask = (attention_mask * disp).squeeze(0).squeeze(0).to(self.device)

    # breakpoint()
    if self.opt.batch_size > 1:
        # 1, 1 192 * batch_size , 640. so paste the images under eachother such that you only have to do edge detection once
        depth_mask = depth_mask.view(self.opt.height * self.opt.batch_size, self.opt.width).to(self.device)

    depth_mask = np.array(depth_mask.cpu().detach().numpy())

    # * 255 for canny edge working
    depth_mask = depth_mask * 255

    # cast to int for correct canny input
    depth_mask = np.uint8(depth_mask)

    # # prepare images for plotting
    if batch_idx % self.opt.save_plot_every == 0 and scale == 0:
        disp_min, disp_max, original_img, disp, path, attention_mask = self.prepare_edge_plot(disp, inputs,
                                                                                              attention_mask)

    edges_disp = cv2.Canny(depth_mask, self.opt.edge_detection_threshold * 255, self.opt.edge_detection_threshold * 255,
                           apertureSize=3, L2gradient=False)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(depth_mask, kernel, iterations=3)
    result = cv2.bitwise_and(edges_disp, edges_disp, mask=erosion)

    for b in range(self.opt.batch_size - 1):
        # remove edges between the stacked images borders
        number = (b + 1) * (self.opt.height)
        result[number] = 0
        result[number - 1] = 0

    # if you found an edge and save the image. only once in the self.opt.save_edge_img steps because of computational speed
    if result.sum() > 0 and batch_idx % self.opt.save_plot_every == 0 and scale == 0:
        # print("SAVEEUU")

        fig, ax = plt.subplots(6, 1, figsize=(12, 12))

        # 2, 192, 640, 3
        ax[0].imshow(original_img)
        ax[0].title.set_text('Original image')
        ax[0].axis('off')
        #
        # # 2, 192, 640
        ax[1].imshow(disp)
        ax[1].title.set_text('disp')
        ax[1].axis('off')

        # ax[2].imshow(original_attention.cpu()[batch][index_nrs_not_overlapping[batch][attention]], cmap='cividis')
        # ax[2].title.set_text('Original attention mask')
        # ax[2].axis('off')

        # # #2, 1, 192, 640
        ax[2].imshow(attention_mask.cpu(), cmap='cividis')
        ax[2].title.set_text('Casted attention mask')
        ax[2].axis('off')



        ax[3].imshow(depth_mask, vmin=disp_min, vmax=disp_max)
        ax[3].title.set_text('depth mask')
        ax[3].axis('off')

        ax[4].imshow(edges_disp)
        ax[4].title.set_text('Edges disp before erosion')
        ax[4].axis('off')

        ax[5].imshow(result)
        ax[5].title.set_text('Edges disp after erosion')
        ax[5].axis('off')

        # plt.show()

        fig.savefig(
            '{}/epoch_{}_batchIDX_{}_result_{}_threshold_{}.png'.format(path, self.epoch, batch_idx, result.sum() / 255,
                                                                        self.opt.edge_detection_threshold))
        plt.close()

    if result.sum() > 0:
        edge_loss.append(result.sum())

    edge_loss = torch.FloatTensor(edge_loss).to(self.device)

    # divide by 255 because 1 pixels should have value 1 not 255
    loss = torch.mean(edge_loss) / 255

    if torch.isnan(loss):
        loss = 0

    return loss

def prepare_edge_plot(self, disp, inputs, attention_mask):
    """
    Once in a while make a plot of the edge loss. Therefor prepare the min and max values for correct depth mask plot
    retreive the original kitti image
    and set correct dimensions for the depth image
    """

    attention_mask = attention_mask.view(self.opt.height * self.opt.batch_size, self.opt.width)

    path = self.opt.log_dir + self.opt.model_name + "/" + "edge_loss_img/"

    # print("EDGE LOSS PATH", path)


    if not os.path.exists(path):
        os.makedirs(path)


    disp = disp.view(self.opt.height * self.opt.batch_size, self.opt.width)

    disp_min = np.uint8(disp.min().cpu().detach().numpy() * 255)
    disp_max = np.uint8(disp.max().cpu().detach().numpy() * 255)

    original_img = inputs["color_aug", 0, 0]



    original_img = original_img.view(self.opt.height * self.opt.batch_size, self.opt.width, 3)
    # depth_mask = depth_mask.view(self.opt.height * self.opt.batch_size, self.opt.width, 3)

    original_img = np.array(original_img.cpu().detach().numpy())


    # original_img = np.swapaxes(original_img, 1, 2)
    # original_img = np.swapaxes(original_img, 0, 1)


    # filter out dimension for correct plotting
    disp = disp.squeeze(1)
    disp = np.array(disp.cpu().detach().numpy())

    return disp_min, disp_max, original_img, disp, path, attention_mask




def additional_not_overlapping_check(self, attention_mask):
    # additional check if the not overlapping masks are not overlapping
    for batch in range(self.opt.batch_size):

        # loop over the attention masks per batch
        for mask in range(attention_mask[batch].shape[0]):

            # these are dummy mask in order to correctly fill the tensor
            if attention_mask[batch][mask].sum() == 0:
                continue

            # create a list with all the mask index number
            masks_to_compare_with = np.arange(0, attention_mask[batch].shape[0])

            # you want to compare the mask x from batch x with the attention_masks y from batch x
            overlap = find_overlapping_tensors(batch, mask, masks_to_compare_with, attention_mask)

            assert len(overlap) == 1, "there are still overlapping values"

