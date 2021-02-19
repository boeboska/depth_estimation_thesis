# for batch in inputs['attention']:
#     for tensor in batch:
#         # None
#         print(tensor.mean())

# breakpoint()

# fig, ax = plt.subplots(2, 1)
#
# target_img = inputs["color_aug", 0, 0].squeeze(0).permute(1, 2, 0).cpu()
#
#
# # breakpoint()
# target_img = transforms.ToPILImage()(target_img)
#
# ax[0].imshow(target_img)
#
#
#
#
# for key in inputs.keys():
#     if key[0] == 'attention':
#
#
#         attention_prob = round(float(key[1].split('_')[1].split('.jpg')[0]), 2)
#
#         attention_mask = inputs[key]
#         attention_mask_save = inputs[key]
#         attention_mask_save = copy.deepcopy(attention_mask_save)
#
#         # everywhing which doesn't belong to the mask, set off
#         attention_mask[attention_mask < 0.8] = 0
#
#         # everything which belongs to the mask, set onn
#         attention_mask[attention_mask >= 0.8] = 1
#
#         sum_mask = round(attention_mask.sum().item())
#         sum_mask = '{:,}'.format(sum_mask)
#         name = str(attention_prob) + "_____" + str(sum_mask)
#
#         # breakpoint()
#         attention_mask_save = attention_mask_save.squeeze(0).squeeze(0)
#
#         img_pil = transforms.ToPILImage()(attention_mask_save)
#
#         # ax[1] = img_pil
#         ax[1].imshow(attention_mask_save)
#         plt.show()
#
#         plt.savefig('sum_attention/test.jpg', dpi = 100)
#         print('--')
# plt.imsave('sum_attention/{}.jpg'.format(name), img_pil, cmap="cividis")
#
#         #  create a dict with prob as key and value is a list of the sum of attention masks
#
#         if attention_prob in prob_sum_mask:
#             current_list = prob_sum_mask[attention_prob]
#             current_list.append(attention_mask.sum().item())
#             prob_sum_mask[attention_prob] = current_list
#
#         else:
#             prob_sum_mask[attention_prob] = [attention_mask.sum().item()]
#
#
# print("BATCH IDX", batch_idx)
# if batch_idx == 100:
#
#     # sort dict
#     prob_sum_mask = collections.OrderedDict(sorted(prob_sum_mask.items()))
#     # convert the list of sum attention mask values to avg and std values
#     for key in prob_sum_mask.keys():
#
#         current_list = np.array(prob_sum_mask[key])
#         mean = np.mean(current_list)
#         std = np.std(current_list)
#
#
#         prob_sum_mask[key] = (mean, std)
#
#     # breakpoint()
#     # prepare for plotting
#     avg_sum = []
#     std_sum = []
#     probability = []
#     for key, value in prob_sum_mask.items():
#
#         probability.append(key) # 0.01 ....
#         avg_sum.append(value[0]) # height
#         std_sum.append(value[1])
#     # breakpoint()
#
#     # breakpoint()
#     y_pos = np.arange(len(probability))
#
#
#     f, ax = plt.subplots(figsize=(18, 5))
#     plt.bar(y_pos, avg_sum, yerr = std_sum)
#
#     plt.xticks(y_pos, probability, rotation = 'vertical')
#
#     plt.ylabel('AVG SUM')
#     plt.xlabel("ATTENTION PROB")
#     plt.title('')
#
#     plt.show()

# print(prob_sum_mask)
