import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import seaborn as sns
import copy
import itertools


def plot_loss_tensor(self, inputs, to_optimise, attention_mask_weight, batch_idx, scale, idxs, identity_reprojection_loss):

    fig, axis = plt.subplots(5, 1, figsize=(20, 20))

    original_img = inputs["color_aug", 0, 0]

    original_img = np.array(original_img.squeeze().cpu().detach().permute(1, 2, 0).numpy())

    axis[0].title.set_text('original kitti image')
    axis[0].axis('off')
    axis[0].imshow(original_img)

    # create the heatmap
    axis[1].title.set_text(f'mean original loss{round(to_optimise.mean().item(),2)}')
    axis[1].axis('off')
    sns.heatmap(to_optimise.cpu().squeeze(0).detach(), ax=axis[1], vmin=0, vmax=0.6, cmap='Greens', center=1)

    axis[2].title.set_text(f'Our weight matrix first{round(attention_mask_weight.mean().item(), 2)}')
    axis[2].axis('off')
    sns.heatmap(attention_mask_weight.cpu().squeeze(0).detach(), ax=axis[2], vmin=0, vmax=1.2, cmap='Greens', center=1)

    # put the original rgb kitti image in the subplot

    # attention_mask_weight[idxs < identity_reprojection_loss.shape[1]] = 1
    #
    # axis[3].title.set_text('Our weight matrix adjusted black holes')
    # axis[3].axis('off')
    # sns.heatmap(attention_mask_weight.cpu().squeeze(0).detach(), ax=axis[3], vmin=1, vmax=1.2, cmap='Greens', center=1)
    # breakpoint()

    # alle pixel waarden tot aan 1.05 map maar 1
    attention_mask_weight[attention_mask_weight <= self.opt.attention_mask_threshold] = self.opt.reduce_attention_weight

    # geef extra weight mee aan pixels die nog over zijn geblevenaa
    attention_mask_weight[attention_mask_weight > 1] = attention_mask_weight[attention_mask_weight > 1] * self.opt.attention_weight


    axis[3].title.set_text(f'Our weight matrix daarna { round(attention_mask_weight.mean().item(), 2)}')
    axis[3].axis('off')
    sns.heatmap(attention_mask_weight.cpu().squeeze(0).detach(), ax=axis[3], vmin=0, vmax=1.2,
                cmap='Greens', center=1)

    to_optimise = to_optimise * attention_mask_weight

    axis[4].title.set_text(f'mean loss after multiplication{round(to_optimise.mean().item(),2)}')
    sns.heatmap(to_optimise.cpu().squeeze(0).detach(), ax=axis[4], vmin=0, vmax=0.6, cmap='Greens', center=1)

    axis[4].axis('off')
    plt.tight_layout()
    fig.savefig(f'loss_tensor_visualization/batch_idx_{batch_idx, scale}.png')
    plt.close(fig)


def determine_weight_matrix(self, overlap_per_pixel, weight_per_mask, attention_masks, method):
    """
    function that receives the an dict which tells which pixels have overlap with which masks
    and what the weight is per mask
    based on the method the function creates the weight matrix
    """

    weight_matrix = torch.ones(size=(attention_masks.shape[0], 192, 640)).to(self.device)
    #     print("WW", weight_per_mask)

    # filter out lists which have the same list values. such that you only have to calculate avg once per unique listf
    unique_keys = {}
    for key, value in overlap_per_pixel.items():
        unique_keys[(key[0], *value)] = [key[0], *value]
    #     test = {key: [(key[0], *value)] for key, value in overlap_per_pixel.items()}
    #     print("@@@", unique_keys)

    unique_overlapping_masks = {}

    # now calculate the weight for the unique keys found in the dict. so the unique combinations of masks
    for u in unique_keys.keys():

        weight = []

        # select the weight mask which corresponds to the batch
        weight_mask = weight_per_mask[u[0]].to(self.device)

        u = list(u)

        # skip first items since we use that for indexing wheight matrix
        for number in u[1:]:
            weight.append(weight_mask[number].item())

        # calculate avg value for the overlapping masks combination
        if method == 'avg':
            unique_overlapping_masks[tuple(u)] = np.average(np.array(weight))

        if method == 'min':
            unique_overlapping_masks[tuple(u)] = np.min(np.array(weight))

        if method == 'max':
            unique_overlapping_masks[tuple(u)] = np.max(np.array(weight))

    sum_weight = 0
    start = time.time()
    for key, value in overlap_per_pixel.items():
        # prepare for correct value indexing in weight dict
        curr = value
        curr = copy.deepcopy(curr)
        curr.insert(0, key[0])

        # print("PLACE", key[0], key[1], key[2])
        # #         # set the avg weight in the matrix. do 1 + the weight otherwise the loss goes down after multiplication
        weight_matrix[key[0], key[1], key[2]] = 1 + unique_overlapping_masks[tuple(curr)]
        sum_weight += 1 + unique_overlapping_masks[tuple(curr)]

    # # #     # check if the weights are correctly multiplied
    check = round(
        int(weight_matrix.sum() - attention_masks.shape[0] * attention_masks.shape[2] * attention_masks.shape[3]) - (
                    sum_weight - len(overlap_per_pixel)), 2)
    # print("check", check)
    # print("@@@", start - time.time())

    assert check < 1, "incorrect weights mulitplied, you are left with: {}".format(check)

    return weight_matrix.to(self.device)


def weight_per_pixel(overlap, weight_per_pixel, index_dict):
    """
    function which receives the overlap per pixel and the combinations.
    it converts these combinations back to the attention mask numbers
    """

    which_pixels_overlap = {}
    for x in overlap:
        #         print("xx", x)
        # check if there is already overlap for that pixel for that batch
        if (x[0].item(), x[2].item(), x[3].item()) in which_pixels_overlap:

            # there are the masks in that batch which have overlap for this pixel
            # sellect the current masks which have overlap for that pixel in that batch
            current_masks_for_pixel = which_pixels_overlap[x[0].item(), x[2].item(), x[3].item()]

            # select the new masks which are having overlap for that pixel
            new_masks = index_dict[x[1].item()]

            # concatenate them
            end_list = current_masks_for_pixel + list(set(new_masks) - set(current_masks_for_pixel))

            # put them back in the dict
            which_pixels_overlap[x[0].item(), x[2].item(), x[3].item()] = end_list



        # there is not yet overlap found for that specific pixel for that batch
        else:
            #
            which_pixels_overlap[x[0].item(), x[2].item(), x[3].item()] = index_dict[x[1].item()]
    return which_pixels_overlap


def check_overlap_per_pixel(self, attention_masks):
    """
    function that check which masks are having overlap per pixel
    """

    # first determine how much wheight each mask gets based on its size
    weight_per_mask = calculate_weight_per_mask(self, attention_masks).to(self.device)

    # make sure that all pixels outside the mask are not similar by giving them other value
    for b in range(attention_masks.shape[0]):
        for a in range(attention_masks.shape[1]):
            attention_masks[b][a][attention_masks[b][a] == 0] = -a - 1

    # make a list with the index numbers of all the attention masks
    all_masks = attention_masks.shape[1]
    all_masks = np.arange(all_masks)

    # make all possible combinations with attention masks
    masks_combinations = list(itertools.combinations((all_masks), 2))

    # create a dict and fill with all the possible combinations
    index_dict = {}
    for i, c in enumerate(masks_combinations):
        index_dict[i] = [c[0], c[1]]

    # create two empty tensors for preparing torch . eq
    first = torch.zeros(
        size=(attention_masks.shape[0], len(masks_combinations), attention_masks.shape[2], attention_masks.shape[3])).to(self.device)
    second = torch.zeros(
        size=(attention_masks.shape[0], len(masks_combinations), attention_masks.shape[2], attention_masks.shape[3])).to(self.device)

    for b in range(attention_masks.shape[0]):
        # fill the empty tensors with the combinations of attention masks
        for i, combination in enumerate(masks_combinations):
            first[b][i] = attention_masks[b][combination[0]].to(self.device)
            second[b][i] = attention_masks[b][combination[1]].to(self.device)

    # calculate which combinations have overlap and where (which pixels)
    overlap = torch.eq(first, second).nonzero().to(self.device)


    #     for row in overlap:
    #         print(row)
    #     print(overlap)

    # find out which masks having overlap. so from the combination convert back to attention mask numbers
    # dictionary which tells per pixel which attention masks are having overlap
    per_pixel_overlap = weight_per_pixel(overlap, weight_per_pixel, index_dict)

    #     print("PER PIXEL OVERLAP ", per_pixel_overlap)
    #     for key, value in per_pixel_overlap.items():
    #         print(key, value)

    return per_pixel_overlap, weight_per_mask


def calculate_weight_per_mask(self, attention_masks):
    # set values again to their normal values. First their were casted to negative such that there was no torch.eq outside the mask
    attention_masks[attention_masks >= 0.8] = 1
    attention_masks[attention_masks < 0.8] = 0

    # batch x amount of attention. sum every attention tensor within the batch size
    attention_sum = attention_masks.sum(-1).sum(-1).to(self.device)

    # within a batch you have multiple attention masks. Decide how much weight each attention mask will receive.
    # The smaller the mask the more weight it'll receive.

    # summ all the attention maps within 1 batch size
    # this is the summ of all the attention maps withing 1 batch
    batch_sum = attention_sum.sum(-1).unsqueeze(1).to(self.device)

    v = attention_sum / (self.opt.height * self.opt.width)

    v = 1 / v
    # remove inf number because 1 / 0 = inf
    v[v == float('inf')] = 0
    v[v != v] = 0

    avg_weight_per_threshold = {
        0.4: 1053,
        0.5: 980,
        0.6: 905,
        0.7: 818,
        0.8: 688
    }

    attention_weight_matrix = v / avg_weight_per_threshold[self.opt.attention_threshold]

    # remove nan
    attention_weight_matrix[attention_weight_matrix != attention_weight_matrix] = 0

    return attention_weight_matrix

def calculate_weight_matrix(self, inputs, batch_idx, original_masks):

    start = time.time()

    attention_masks = inputs['attention'].to(self.device)

    # overlap per pixel is a dictionary telling you per pixel which attention masks have overlap there
    # weight per mask is an array of len amount attetnion masks. each number tells how much wheight that attention mask is
    # 2 sec
    overlap_per_pixel, weight_per_mask = check_overlap_per_pixel(self, attention_masks)
    # print("START", start)
    # time_overlap_per_pixel =  start - time.time()

    # print("TIME OVERLAP PER PIXEL", time_overlap_per_pixel)

    # now create the weight matrix
    # 2 sec
    weight_matrix =  determine_weight_matrix(self, overlap_per_pixel, weight_per_mask , attention_masks, method = 'avg').to(self.device)

    time_determine_weight_matrix =  start - time.time()

    # print("time weight matrix", time_determine_weight_matrix)


    # this one is quick 0.1 sec
    weight_matrix = determine_not_overlapping_masks(self, weight_matrix, attention_masks, weight_per_mask).to(self.device)

    time_determine_not_overlap_pixel =  start - time.time()
    if batch_idx % self.opt.save_plot_every == 0:
    # print("determine not overlap masks", time_determine_not_overlap_pixel)
        plot_attention_weight_loss_matrix(self, inputs, attention_masks, original_masks, weight_per_mask, batch_idx, weight_matrix)

    end = time.time() - start
    print("TOTAL", end)

    return weight_matrix


def determine_not_overlapping_masks(self, weight_matrix, attention_masks, weight_per_mask):
    """
    Function checks which pixels doesn't overlap between attention masks and applies
    the weight from these attention mask on the weight matrix on the not overlapping pixels
    """

    # all values which are 1 in the weight matrix
    weight_matrix[weight_matrix == 1] = 999

    # also set mask values on 999
    # now if you find overlap between the two then you know that that pixel has not been overlap found for
    attention_masks[attention_masks >= 0.8] = 999
    attention_masks[attention_masks < 0.8] = 0

    # create two empty tensors for preparing torch . eq
    first = torch.zeros(
        size=(attention_masks.shape[0], attention_masks.shape[1], attention_masks.shape[2], attention_masks.shape[3])).to(self.device)
    second = torch.zeros(
        size=(attention_masks.shape[0], attention_masks.shape[1], attention_masks.shape[2], attention_masks.shape[3])).to(self.device)

    # comare every mask with the weight matrix to find not overlapping pixels
    for b in range(attention_masks.shape[0]):
        for mask in range(attention_masks.shape[1]):
            first[b][mask] = attention_masks[b][mask].to(self.device)
            second[b][mask] = weight_matrix[b].to(self.device)

    # this is a tensor telling which mask have no overlap with other masks. so they can add their weight to the matrix without avg, min , max, etc
    overlapping = torch.eq(first, second).to(self.device)

    # breakpoint()



    for batch in range(overlapping.shape[0]):
        for overlap_mask in range(overlapping.shape[1]):
            # check how many not overlapping pixels there are for this mask
            amount_true_values = overlapping[batch][overlap_mask].sum().item()

            # get the weight value for this mask
            weight = weight_per_mask[batch][overlap_mask].to(self.device)

            # fill a tensor with the masks weight value + 1 because otherwise the ssim, l1 loss will shrink because the weight is < 1
            weights = torch.full(size=(1, amount_true_values), fill_value= 1 + weight).to(self.device)

            # fill the weight matrix with the weight of the not overlapping mask pixels
            weight_matrix[batch][overlapping[batch][overlap_mask]] = weights

    # set to zero such that the loss (SSIM, L1) gets multiplied with 1. in this way the loss doensn't get lost
    weight_matrix[weight_matrix == 999] = 1

    return weight_matrix


def plot_attention_weight_loss_matrix(self, inputs, attention_masks, original_attention_masks, weight_per_mask, batch_idx, weight_matrix):
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
    if not os.path.exists(path):
        os.makedirs(path)

    attention_masks[attention_masks >= 0.8] = 1
    attention_masks[attention_masks < 0.8] = 0

    for batch_nr in range(self.opt.batch_size):


        # select the rgb kitti image
        # select only the first batch
        original_img = inputs["color_aug", 0, 0][batch_nr]
        original_img = np.array(original_img.cpu().detach().numpy())

        original_img = np.swapaxes(original_img, 0, 1)
        original_img = np.swapaxes(original_img, 1, 2)




# count how many non zero attention masks there are
        plot_list = []
        for mask in range(attention_masks.shape[1]):
            current_mask = attention_masks[batch_nr][mask]
            if current_mask.sum() != 0:
                plot_list.append(current_mask)



        # print(len(plot_list))
        # breakpoint()
        # determine how many rows there should come in the subplot based on the amount of non-overlapping attention masks
        if len(plot_list) / 2 != 0:
            amount_sub_rows = int((len(plot_list) + 1) / 2)
        else:
            amount_sub_rows = int(len(plot_list) / 2)


        fig, axis = plt.subplots(2, 2, figsize=(20, 5))

        # breakpoint()

            # create the heatmap
        sns.heatmap(weight_matrix[batch_nr].cpu(), ax=axis[0, 0], vmin=1, vmax=1.2, cmap='Greens', center=1)
        sns.heatmap(weight_matrix[batch_nr].cpu(), ax=axis[1, 0], vmin=1, vmax=1.05, cmap='Greens', center=1)
        sns.heatmap(weight_matrix[batch_nr].cpu(), ax=axis[1, 1], vmin=1, vmax=1.01, cmap='Greens', center=1)

        # put the original rgb kitti image in the subplot
        axis[0, 1].imshow(original_img)

        fig.savefig('{}/epoch_{}_batch_{}_batch_idx_{}_p1.png'.format(path, self.epoch, batch_nr, batch_idx))
        plt.close(fig)


        fig, axis = plt.subplots(amount_sub_rows, 2, figsize=(12, 12))
        for mask in range(len(plot_list)):

            # breakpoint()

            # determine where to put the attention mask in the subplot
            axis[int(np.floor(mask / 2))][mask % 2].imshow(original_attention_masks[batch_nr][mask].cpu().numpy(),
                                                           cmap='cividis')

            # breakpoint()
            # the title is the corresponding mask weight
            axis[int(np.floor(mask / 2))][mask % 2].title.set_text(
                'Weight:{}'.format(round(weight_per_mask[batch_nr][mask].item(), 3)))

            # set axis off
            axis[int(np.floor(mask / 2))][mask % 2].axis('off')

        fig.savefig('{}/epoch_{}_batch_{}_batch_idx_{}_p2.png'.format(path, self.epoch, batch_nr, batch_idx))
        plt.close(fig)