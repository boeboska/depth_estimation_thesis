import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import seaborn as sns


def select_non_zero_attention_masks(attention_masks):
    """
    This function checks how many attention masks there are per batch
    It also maps pixels outside the mask to negative values. Every mask will receive a different negative value
    such that there will be no overlap between the masks
    """

    amount_attention_masks = {}

    # create an empty dictionary where you save what the size is of every attention masks. this will be used later on
    # for determening the weight per attention mask
    size_attention_masks_all_batches = {}
    size_per_batch = {}

    for b in range(attention_masks.shape[0]):
        amount_attention_masks[b] = None

        size_attention_masks_all_batches[b] = None

        for m in range(attention_masks.shape[1]):
            size_per_batch[m] = None

        size_attention_masks_all_batches[b] = size_per_batch



    # loop over batches
    for b in range(attention_masks.shape[0]):



        # loop over the attention masks per batch
        for mask in range(attention_masks.shape[1]):

            # then there is no attention mask found for this image
            if attention_masks[b][mask].sum() == 0:

                # add inf such that when you do sorting later on these values will be add the end after sorting
                size_attention_masks_all_batches[b][mask] = float('inf')

                # only add the first number per batch because then the oter number from there are also no attention masks
                # this is the number of how many attention masks there are found per batch
                if amount_attention_masks[b] == None:
                    amount_attention_masks[b] = mask

                attention_masks[b][mask][attention_masks[b][mask] == 0] = -999

            else:

                # add the size of the attention mask to the dictionary
                size_attention_masks_all_batches[b][mask] = attention_masks[b][mask].sum().item()

                attention_masks[b][mask][attention_masks[b][mask] == 0] = -mask - 1

    return attention_masks, amount_attention_masks, size_attention_masks_all_batches

def find_not_overlapping_masks(self, inputs, attention_masks, amount_attention_masks, batch_masks, original_attention_masks, batch_idx):
    """
    inputs: used for plotting
    attention_masks: the update attention masks. where values outside the attention masks are mapped to different negative value per mask such that
    there will be no overlap between pixels outside attention masks
    amount_attention_masks: dictionary telling how much attention masks there are per batch
    batch_masks: tensor where we will store the not overlapping attention masks
    original_attention_masks: used for plotting.
    batch_idx: used for plotting
    """

    all_not_overlapping_masks = []

    # loop over the batches
    for batch in range(attention_masks.shape[0]):

        # create a list of how many attention masks there are per batch
        if amount_attention_masks[batch] == None:
            # if none is found, then all attention masks are loaded in
            all_masks_loop_list = np.arange(0, attention_masks[batch].shape[0])

        else:
            all_masks_loop_list = np.arange(0, amount_attention_masks[batch])

        not_overlapping_masks = []

        # while there are still masks not checked if they over overlapping
        while len(all_masks_loop_list) > 0:

            # pick a random mask
            choice_mask = np.random.choice(all_masks_loop_list)

            # select the mask based on the choice
            current_mask = attention_masks[batch][choice_mask]

            # all masks which are available where you want the current mask to compare with
            other_masks = all_masks_loop_list

            # find which masks are all overlapping which current mask (choice_mask).
            overlapping_masks = find_overlapping_tensors(batch, choice_mask, other_masks, attention_masks)

            # add the current mask to the not overlapping mask
            not_overlapping_masks.append(choice_mask)

            # remove all the overlapping masks from the possible masks since they overlap
            all_masks_loop_list = np.setdiff1d(all_masks_loop_list, overlapping_masks)

            # remove also the current mask since you have checked thisone
            all_masks_loop_list = np.setdiff1d(all_masks_loop_list, choice_mask)

        # fill the batch mask array with the not overlapping masks
        for f in range(len(not_overlapping_masks)):
            batch_masks[batch][f] = attention_masks[batch][not_overlapping_masks[f]]


        all_not_overlapping_masks.append(not_overlapping_masks)


    return batch_masks, all_not_overlapping_masks


def plot_attention_weight_loss_matrix(self, inputs, not_overlapping_masks, attention_masks, original_attention_masks, weight_per_mask, batch_idx, weight_assert):
    """
    inputs: for selecting the original kitti image
    not_overlapping_masks: list of index number where the masks not overlap
    attention_masks: the tensor where the attention masks are in
    original_attention_masks: untouched attention masks used for plotting
    weight_per_mask: used for plot title
    batch_idx: used for plot title
    weight_assert:  used for plot title
    """


    path = self.opt.log_dir + self.opt.model_name + "/" + "weight_matrix_img/"
    print("PATHH", path)
    if not os.path.exists(path):
        os.makedirs(path)

    # select the rgb kitti image
    original_img = inputs["color_aug", 0, 0]
    original_img = np.array(original_img.cpu().detach().numpy())

    plot_list = []
    for mask in range(len(not_overlapping_masks)):
        if attention_masks[0][not_overlapping_masks[mask]].sum() == 0:
            continue
        else:
            plot_list.append(not_overlapping_masks[mask])

    # determine how many rows there should come in the subplot based on the amount of non-overlapping attention masks
    if len(plot_list) / 2 != 0:
        amount_sub_rows = int((len(plot_list) + 1) / 2)
    else:
        amount_sub_rows = int(len(plot_list) / 2)

    # set the super negative values to zero for correct plotting
    attention_masks[attention_masks < 0.8] = 0

    for heat_mask in weight_per_mask:

        fig, axis = plt.subplots(1, 2, figsize=(40, 5))

        # create the heatmap
        sns.heatmap(heat_mask.squeeze(0), ax=axis[0], vmin=1, vmax=1.2, cmap='Greens', center=1)

        # put the original rgb kitti image in the subplot
        axis[1].imshow(original_img[0][1])



    fig.savefig('{}/epoch_{}_batch_idx_{}_p1.png'.format(path, self.epoch, batch_idx))
    plt.close(fig)

    fig, axis = plt.subplots(amount_sub_rows, 2, figsize=(12, 12))

    for mask in range(len(plot_list)):

        # determine where to put the attention mask in the subplot
        axis[int(np.floor(mask / 2))][mask % 2].imshow(original_attention_masks[0][plot_list[mask]].cpu().numpy(),
                                                       cmap='cividis')
        # the title is the corresponding mask weight
        axis[int(np.floor(mask / 2))][mask % 2].title.set_text(
            'Weight:{}'.format(round(weight_assert[0][mask].item(), 3)))
        axis[int(np.floor(mask / 2))][mask % 2].axis('off')

    fig.savefig('{}/epoch_{}_batch_idx_{}_p2.png'.format(path, self.epoch, batch_idx))
    plt.close(fig)

def check_if_weight_is_applied_correctly(weight_per_mask, weight_assert):
    """
    weight_per_mask: tensor of shape: [batch, 1, img_width, img_height]. It is thus 1 tensor where all the weight are
    in. This tensor will be multiplied againt the SSIM + L1 loss tensor.
    weight_assert: [batch, amount_of_masks] telling how much weight each attention mask should get in each batch.
    The goal of the function is to check if the weights from weight_assert are correctly incorporated in the weight_per_mask
    # it check if is finds the weight numbers from weight assert back in the weight_per_mask tensoer
    """

    # code to check if multiplicatiton was correct
    weight_after_multiplication = []

    # loop over the masks per batch
    for b in range(weight_per_mask.shape[0]):
        for mask in range(weight_per_mask.shape[1]):

            # calclate the unique values of the mask by ignoring the 1 values. Because the 1 values didn't
            # get any additional weight from an attention mask
            weights_in_tensor = weight_per_mask[b][mask][weight_per_mask[b][mask] != 1.].unique()

            # sum of all the unique values in the tensor. these are the additional weight values
            weights_in_tensor = (weights_in_tensor - 1).sum().item()

            # round for floating overflow. weight assert are the values which should be used.
            # take the unique value in weight asser ass well because if 2 masks get same value check doesn't hold but is still works correctly
            check = round( round(weights_in_tensor, 2) - (weight_assert.unique().sum().item()), 2)

            # check should be zero, otherwise throw the error message
            assert check == 0, "Weight are not correctly multipled. The weight different is: {}".format(check)


def overlapping_masks_edge_detection(self, inputs, batch_idx, original_attention_masks):

    # retrieve the attention masks which belong to the target frame
    attention_masks = inputs['attention'].to(self.device)

    # only keep the pixels which belong to the mask en reduce noise from the mask image
    attention_masks[attention_masks >= 0.8] = 1
    attention_masks[attention_masks < 0.8] = 0

    # checks how many attention masks there are per batch
    # It also maps pixels outside the mask to negative values. Every mask will receive a different negative value
    # such that there will be no overlap between the masks
    attention_masks, amount_attention_masks, attention_size_dict = select_non_zero_attention_masks(attention_masks)
    # attention masks are the updated masks.
    # amount attention masks is a dict telling how many attention masks there are per batch

    # empty tensor which will be filled later on with the not overlapping masks
    batch_masks = torch.zeros(size=(attention_masks.shape[0], attention_masks.shape[1], 192, 640))

    batch_masks, index_nrs_all_not_overlapping_masks = find_not_overlapping_masks_for_edge_detection(self, inputs, attention_masks, amount_attention_masks, batch_masks,
                                                  original_attention_masks, batch_idx, attention_size_dict)

    return batch_masks, index_nrs_all_not_overlapping_masks




def find_not_overlapping_masks_for_edge_detection(self, inputs, attention_masks, amount_attention_masks, batch_masks, original_attention_masks, batch_idx, attention_size_dict):
    """
    inputs: used for plotting
    attention_masks: the update attention masks. where values outside the attention masks are mapped to different negative value per mask such that
    there will be no overlap between pixels outside attention masks
    amount_attention_masks: dictionary telling how much attention masks there are per batch
    batch_masks: tensor where we will store the not overlapping attention masks
    original_attention_masks: used for plotting.
    batch_idx: used for plotting
    """

    # sort attention_size_dict on the size of the attention masks
    for batch in range(len(attention_size_dict)):
        # sort the dict per batch on the size. based on smallest size first
        attention_size_dict[batch] = {k: v for k, v in sorted(attention_size_dict[batch].items(), key=lambda item: item[1])}

    # print("DICHT", attention_size_dict)
    all_not_overlapping_masks = []

    # loop over the batches
    for batch in range(attention_masks.shape[0]):

        all_masks_loop_list = []


        # convert the sorted dict to a list and filter out the INF values since they are not masks
        for key, value in attention_size_dict[batch].items():

            if str(value) != 'inf':
                all_masks_loop_list.append(key)

        not_overlapping_masks = []

        # while there are still masks not checked if they over overlapping
        while len(all_masks_loop_list) > 0:

            # select the first item of the all masks loop list since that is the smallest attention mask
            choice_mask = all_masks_loop_list[0]

            # all masks which are available where you want the current mask to compare with
            other_masks = all_masks_loop_list

            # find which masks are all overlapping which current mask (choice_mask).
            overlapping_masks = find_overlapping_tensors(batch, choice_mask, other_masks, attention_masks)

            # add the current mask to the not overlapping mask
            not_overlapping_masks.append(choice_mask)

            # remove all the overlapping masks from the possible masks since they overlap
            all_masks_loop_list = np.setdiff1d(all_masks_loop_list, overlapping_masks)

            # remove also the current mask since you have checked thisone
            all_masks_loop_list = np.setdiff1d(all_masks_loop_list, choice_mask)


        # fill the batch mask array with the not overlapping masks
        for f in range(len(not_overlapping_masks)):
            batch_masks[batch][f] = attention_masks[batch][not_overlapping_masks[f]]

        all_not_overlapping_masks.append(not_overlapping_masks)

    return batch_masks, all_not_overlapping_masks


def attention_mask_weight(self, inputs, batch_idx, original_attention_masks, edge=False):
    """
    inputs: for selecting the attention masks and original kitti image for plotting
    batch_idx: used for saving the attention weight matrix image
    original_attention: original attention masks used for plotting
    Goal of the function: The function received attention masks. it filters out overlapping attention masks
    and it determines how much weight each attention mask should get based on their size
    """

    # retrieve the attention masks which belong to the target frame
    attention_masks = inputs['attention']

    # only keep the pixels which belong to the mask en reduce noise from the mask image
    attention_masks[attention_masks >= 0.8] = 1
    attention_masks[attention_masks < 0.8] = 0


    # checks how many attention masks there are per batch
    # It also maps pixels outside the mask to negative values. Every mask will receive a different negative value
    # such that there will be no overlap between the masks
    attention_masks, amount_attention_masks, attention_size_dict = select_non_zero_attention_masks(attention_masks)
    # attention masks are the updated masks.
    # amount attention masks is a dict telling how many attention masks there are per batch


    # empty tensor which will be filled later on with the not overlapping masks
    batch_masks = torch.zeros(size=(attention_masks.shape[0], attention_masks.shape[1], 192, 640))

    # checks which masks are overlapping and returns the not overlapping tensors in batch masks.
    # all_not_overlapping_masks are the index numbers of the not overlapping masks
    batch_masks, all_not_overlapping_masks = find_not_overlapping_masks(self, inputs, attention_masks, amount_attention_masks, batch_masks, original_attention_masks, batch_idx)

        # batch masks is de tensor where per batch the not overlapping tensors are filled in.
        #
        # -25., -25., -25., ..., -25., -25., -25.],
        # [-25., -25., -25., ..., -25., -25., -25.],
        # ...,
        # [-25., -25., -25., ..., -25., -25., -25.],
        # [-25., -25., -25., ..., -25., -25., -25.],
        # [-25., -25., -25., ..., -25., -25., -25.]],
        #
        # [[-16., -16., -16., ..., -16., -16., -16.],
        #  [-16., -16., -16., ..., -16., -16., -16.],
        #  [-16., -16., -16., ..., -16., -16., -16.],
        #  ...,
        #  [-16., -16., -16., ..., -16., -16., -16.],
        #  [-16., -16., -16., ..., -16., -16., -16.],
        #  [-16., -16., -16., ..., -16., -16., -16.]],
        #
        # [[-27., -27., -27., ..., -27., -27., -27.],
        #  [-27., -27., -27., ..., -27., -27., -27.],
        #  [-27., -27., -27., ..., -27., -27., -27.],
        #  ...,
        #  [-27., -27., -27., ..., -27., -27., -27.],
        #  [-27., -27., -27., ..., -27., -27., -27.],
        #  [-27., -27., -27., ..., -27., -27., -27.]],
        #
        # ...,
        #
        # [[0., 0., 0., ..., 0., 0., 0.],
        #  [0., 0., 0., ..., 0., 0., 0.],
        #  [0., 0., 0., ..., 0., 0., 0.],
        #  ...,
        #  [0., 0., 0., ..., 0., 0., 0.],
        #  [0., 0., 0., ..., 0., 0., 0.],
        #  [0., 0., 0., ..., 0., 0., 0.]],
        #
        # [[0., 0., 0., ..., 0., 0., 0.],
        #  [0., 0., 0., ..., 0., 0., 0.



    # additional check if the not overlapping masks are not overlapping
    #     for mask in not_overlapping_masks:
    #         overlap = find_overlapping_tensors(batch, mask, not_overlapping_masks)
    #         assert len(overlap) == 1, "there are still overlapping values"

    # if you train with edge loss, you are only interested in which masks are not overlapping and not interested in the weights
    # if edge loss is activated return the not overlapping masks. you don't have to calculate a weight per mask
    if edge == True:
        return batch_masks, all_not_overlapping_masks

    # determine weight per mask based on mask size
    weight_per_mask, weight_assert = determine_mask_weight(self, batch_masks) # 0.15 sec for batch size 2
    # mask per weight is 1 tensor per batch where the weight are in. this will be multiplied later in against the SSIM and L1 norm
    # weight_assrt is a tensor per batch with the weight numbers

    # plot every x steps an example of the weight matrix during training
    if batch_idx % self.opt.save_plot_every == 0:
        # only plot for one image .. so therefor the [0] index in all not overlapping masks
        plot_attention_weight_loss_matrix(self, inputs, all_not_overlapping_masks[0], attention_masks,
                                          original_attention_masks, weight_per_mask, batch_idx, weight_assert)


    # check if the weight are correctly multiplied against the attention_masks
    check_if_weight_is_applied_correctly(weight_per_mask, weight_assert)

    return weight_per_mask

def find_overlapping_tensors(batch, current_tensor, other_tensors, attention_masks):
    """
    batch: INTEGER: current batch number
    current_tensor: INTEGER: 1 specific tensor which you want to compare with other tensors
    other_tensors: LIST OF INTEGERS: these are the tensor which will be compared with the current tensor
    # attention_masks: tensor where the attention_masks are putted in.
    The function checks if current tensor is having overlap with other tensors. It uses the batch, current_tensor
    and other_tensors for indexing into the attention_masks
    """

    # build tensors where all the tensors can fit in such that you only have to do torch.eq once
    a = torch.zeros(size = (len(other_tensors), 192, 640))
    b = torch.zeros(size = (len(other_tensors), 192, 640))

    # fill the tensors with the masks
    for x in range(len(other_tensors)):

        # stack one tensor with only the current tensor
        a[x] = attention_masks[batch][current_tensor]

        # stack other tensor with the tensors you want the current tensor to compare with
        b[x] = attention_masks[batch][other_tensors[x]]

    # compare
    result_torch_eq = torch.eq(a, b)

    # count how many pixels ovelap there is between the two images
    count_overlap_pixel = (result_torch_eq == True).nonzero()

    # find which masks are overlapping (which mask numbers)
    overlappers = np.array(count_overlap_pixel[:,0].unique())
    other_tensors = np.array(other_tensors)

    # select correct overlapping tensors
    overlapping_tensors = other_tensors[overlappers]

    return overlapping_tensors


def determine_mask_weight(self, attention_masks):
    """
    This function receives the attention masks. Is calculated a weight per attention mask based on the size of the
    attention mask. The weight is determined by dividing the size of the attention mask by the original image.
    The smaller the attention mask the more weight is will receive. The sum of weights of attention masks within 1 batch
    is 1
    """

    # set values again to their normal values. First their were casted to negative such that there was no torch.eq outside the mask
    attention_masks[attention_masks < 0.8] = 0

    # batch x amount of attention. sum every attention tensor within the batch size. Sum per attention
    attention_sum = attention_masks.sum(-1).sum(-1)


    # divide sum per mask over the size of the kitti image
    v = attention_sum / (self.opt.width * self.opt.height)
    print("attention sum", attention_sum)


    v = 1 / v
    # remove inf number because 1 / 0 = inf
    v[v == float('inf')] = 0
    v[v != v] = 0

    # # determine the weight scale based on the attention masks per batch
    # t = v.sum(-1).unsqueeze(1)

    # now divide the weight over the attention masks based on their size. the smaller the mask the bigger the weight
    attention_weight_matrix = v / (self.opt.width * self.opt.height)



    print("WEIGHTS", attention_weight_matrix)

    # scale the weights by a hyper paramater. in order to not let the weight get to big.
    attention_weight_matrix = attention_weight_matrix * self.opt.weight_attention_matrix

    attention_weight_for_assert = attention_weight_matrix


    # remove nan
    attention_weight_matrix[attention_weight_matrix != attention_weight_matrix] = 0

    # assert round(attention_weight_matrix.sum(-1).sum(-1).item(), 2) == self.opt.weight_attention_matrix , "The sum weight doesn't sum up to {} per batch, namely:{}".format(self.opt.weight_attention_matrix, attention_weight_matrix.sum(-1).sum(-1))

    # how much weight every attention mask gets
    attention_weight_matrix = attention_weight_matrix.unsqueeze(-1).unsqueeze(-1)


    # originam attention masks multiplied by the mask per weight
    # pixels within the attention masks are already on 1, outside mask is zero
    weight_attention_mask = attention_masks * attention_weight_matrix
# weight_attention_mask", weight_attention_mask)

    # [batch, 1, 192, 640]
    end_weight = torch.zeros(attention_masks.shape[0], 1, attention_masks.shape[2], attention_masks.shape[3])

    # loop over batches and create per batch an torch . ones
    for b in range(attention_masks.shape[0]):
        ones = torch.ones(size=(attention_masks.shape[2], attention_masks.shape[3]))

        # now loop over the masks within the batch
        for mask in range(attention_masks.shape[1]):
            # multiply the weights of the current weight mask by an empty ones
            # in this way we will get 1 matrix combines of all the weights
            current_ones = ones * weight_attention_mask[b][mask]
            ones = ones + current_ones

        end_weight[b] = ones

    return end_weight, attention_weight_for_assert
