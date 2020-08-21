
import cv2
import seaborn as sns
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from matplotlib import colors
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

location = "New_York"

# Load the prediction
prediction = cv2.imread(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{location}.png")
prediction = np.array(prediction)
prediction = prediction[:, :, -1]
print(prediction.shape)

prediction[prediction == 217] = 0
prediction[prediction == 49] = 1

plt.imshow(prediction, cmap="tab20c")
plt.title("Prediction")
plt.show()


# Load the ground truth
GT = cv2.imread(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{location}_GT.png")
GT = np.array(GT)
GT = GT[:, :, -1]

# Set to binary mask
GT[GT == 255] = 1


plt.imshow(GT, cmap="tab20c")
plt.title("Ground Truth")
plt.show()


# Calculate Metrics
print(" ")
print("MIoU: ", mean_IU(eval_segm=prediction, gt_segm=GT))
print("ma: ", mean_accuracy(eval_segm=prediction, gt_segm=GT))
print("pa: ", pixel_accuracy(eval_segm=prediction, gt_segm=GT))
print(" ")
print("Frequency Weighted: ", frequency_weighted_IU(eval_segm=GT, gt_segm=prediction))
print("F1 score (macro): ", f1_score(GT, prediction, average="macro"))
print("F1 score (micro): ", f1_score(GT, prediction, average="micro"))
print("F1 score (weighted): ", f1_score(GT, prediction, average="weighted"))
print(" ")

# Plotting
GT[GT == 1] = 2
prediction[prediction == 1] = 3

# Combine the masks
comb = GT + prediction
sns.distplot(comb)
plt.show()



# Count values for a confusion matrix
print("True Negatives", np.count_nonzero(comb == 0))
print("False Positives", np.count_nonzero(comb == 1))
print("False Negatives", np.count_nonzero(comb == 2))
print("True Positives", np.count_nonzero(comb == 3))

TN = np.count_nonzero(comb == 0)
FN = np.count_nonzero(comb == 1)
FP = np.count_nonzero(comb == 2)
TP = np.count_nonzero(comb == 3)


# Plot the matric
cmap = colors.ListedColormap(['darkslategray',   # True Negative
                              'lightcoral',      # False Negative
                              'orange',          # False Positive
                              'cadetblue'])      # True positive

bounds=[0, 1, 2, 3]
norm = colors.BoundaryNorm(bounds, cmap.N)
plt.title("Combined")
plt.show()

plt.imsave(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/"
           f"{location}/{location}_FineTune_DeepLabV3.png",
           comb,
           cmap=cmap)
