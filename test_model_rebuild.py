import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
#!/usr/bin/python

'''
Martin Kersner, m.kersner@gmail.com
2015/11/30
Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.
'''

import numpy as np


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)



# Load the trained model
model = torch.load(f'/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/DeepLabV3-Urban_Water_detection/Data/output/DeepLabV3_epoch_20_New_York_Fine_Tune_batchsize20_it8.pt', map_location='cpu')

# Set the model to evaluate mode
model.eval()

tiles = []
miou_scores = []

# New york: 63

test_location = "New_York"

for i in range(64):

    location = "Osaka"
    region = 5

    if location == "Florida" and region == 1:
        region = ""



    # img = cv2.imread(f"/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/DeepLabV3-Urban_Water_detection/Data/Florida_data/Images/{location}_Region_{region}_{i}_TC_RAW.png")
    # img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    # mask = cv2.imread(f"/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/DeepLabV3-Urban_Water_detection/Data/Florida_data/Masks/{location}_Region_{region}_{i}_Mask_RAW.png")

    # Test path
    img = cv2.imread(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{test_location}/{test_location}_Test_Tiles/{test_location}_{i}_TC.png")
    img = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    # mask = cv2.imread(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{test_location}/{test_location}_Test_Tiles/{test_location}_{i}_Mask.png")

    with torch.no_grad():
        a = model(torch.from_numpy(img).type(torch.FloatTensor)/255)

        # Plot the input image, ground truth and the predicted output
        # plt.figure(figsize=(10,10));
        # plt.subplot(131);
        # plt.imshow(img[0,...].transpose(1,2,0));
        # plt.title('Image')
        # plt.axis('off');
        # plt.subplot(132);
        # plt.imshow(mask);
        # plt.title('Ground Truth')
        # plt.axis('off');
        # plt.subplot(133);
        # plt.imshow(a['out'].cpu().detach().numpy()[0][0]);
        tiles.append(a['out'].cpu().detach().numpy()[0][0]<0.2)
        # plt.imsave(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/here/output/segmentation_output/{i}.png", a['out'].cpu().detach().numpy()[0][0]<0.2, cmap="tab20c")
        # plt.title('Segmentation Output')
        # plt.axis('off');
        # # plt.savefig(f'/Users/tj/PycharmProjects/eval/semantic-segmentation/here/output/segmentation_output/{location}_Region_{region}_{i}.png', bbox_inches='tight')
        print(f"tile: {i}")
        b = a['out'].cpu().detach().numpy()[0][0]<0.2
        b[b == False] = 0
        b[b == True] = 1
        # miou_scores.append((b, mask[:, :, 0]))

print(len(tiles))
print(miou_scores)

if test_location == "New_York":
    row_1 = np.vstack((tiles[0], tiles[1], tiles[2], tiles[3], tiles[4], tiles[5], tiles[6], tiles[7]))
    row_2 = np.vstack((tiles[8], tiles[9], tiles[10], tiles[11], tiles[12], tiles[13], tiles[14], tiles[15]))
    row_3 = np.vstack((tiles[16], tiles[17], tiles[18], tiles[19], tiles[20], tiles[21], tiles[22], tiles[23]))
    row_4 = np.vstack((tiles[24], tiles[25], tiles[26], tiles[27], tiles[28], tiles[29], tiles[30], tiles[31]))
    row_5 = np.vstack((tiles[32], tiles[33], tiles[34], tiles[35], tiles[36], tiles[37], tiles[38], tiles[39]))
    row_6 = np.vstack((tiles[40], tiles[41], tiles[42], tiles[43], tiles[44], tiles[45], tiles[46], tiles[47]))
    row_7 = np.vstack((tiles[48], tiles[49], tiles[50], tiles[51], tiles[52], tiles[53], tiles[54], tiles[55]))
    row_8 = np.vstack((tiles[56], tiles[57], tiles[58], tiles[59], tiles[60], tiles[61], tiles[62], tiles[63]))
    recon = np.hstack((row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8))

else:
    row_1 = np.vstack((tiles[0], tiles[1], tiles[2], tiles[3], tiles[4], tiles[5], tiles[6], tiles[7], tiles[8]))
    row_2 = np.vstack((tiles[9], tiles[10], tiles[11], tiles[12], tiles[13], tiles[14], tiles[15], tiles[16], tiles[17]))
    row_3 = np.vstack((tiles[18], tiles[19], tiles[20], tiles[21], tiles[22], tiles[23], tiles[24], tiles[25], tiles[26]))
    row_4 = np.vstack((tiles[27], tiles[28], tiles[29], tiles[30], tiles[31], tiles[32], tiles[33], tiles[34], tiles[35]))
    row_5 = np.vstack((tiles[36], tiles[37], tiles[38], tiles[39], tiles[40], tiles[41], tiles[42], tiles[43], tiles[44]))
    row_6 = np.vstack((tiles[45], tiles[46], tiles[47], tiles[48], tiles[49], tiles[50], tiles[51], tiles[52], tiles[53]))
    row_7 = np.vstack((tiles[54], tiles[55], tiles[56], tiles[57], tiles[58], tiles[59], tiles[60], tiles[61], tiles[62]))
    row_8 = np.vstack((tiles[63], tiles[64], tiles[65], tiles[66], tiles[67], tiles[68], tiles[69], tiles[70], tiles[71]))
    # row_9 = np.vstack((tiles[72], tiles[73], tiles[74], tiles[75], tiles[76], tiles[77], tiles[78], tiles[79], tiles[80]))
    recon = np.hstack((row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8))

plt.imsave(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{test_location}.png", recon, cmap="tab20c")
