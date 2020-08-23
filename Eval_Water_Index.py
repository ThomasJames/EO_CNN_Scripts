import cv2
import numpy as np
import metrics
import matplotlib.pyplot as plt
from metrics import *
import seaborn as sns
import cv2
import seaborn as sns
from sklearn.metrics import f1_score, precision_score
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from metrics import mean_accuracy, pixel_accuracy, mean_IU


def crop(array_to_crop, array_to_match):
    dims_to_match = array_to_match.shape
    return array_to_crop[0:dims_to_match[0], 0:dims_to_match[1]]


def mean(list_of_items):
    return sum(list_of_items) / len(list_of_items)


def evaluate(data, ground_truth, thresh, index_name):
    print(f"Evaluating {location}, {index_name}")
    data[data > thresh] = 1
    data[data < thresh] = 0

    print("MIoU: ", round(mean_IU(eval_segm=data, gt_segm=ground_truth), 2))
    print("ma: ", round(mean_accuracy(eval_segm=data, gt_segm=ground_truth), 2))
    print("pa: ", round(pixel_accuracy(eval_segm=GT, gt_segm=ground_truth), 2))
    print("F1 score (None): ", round(mean(f1_score(data, ground_truth, average=None)), 2))

    ground_truth[ground_truth == 1] = 2

    comb = ground_truth + data

    TN = np.count_nonzero(comb == 0)
    FN = np.count_nonzero(comb == 2)
    FP = np.count_nonzero(comb == 3)
    TP = np.count_nonzero(comb == 5)

    print("Rate of false positives: ", round(FP / (FP + TN), 2))
    print("Rate of false Negatives: ", round(FN / (FN + TP), 2))

    cmap = colors.ListedColormap(['darkslategray',  # True Negative
                                  'lightcoral',  # False Negative
                                  'orange',  # False Positive
                                  'cadetblue'])  # True positive

    bounds = [0, 2, 3, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imsave(f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{location}/{location}_{index_name}.png",
               comb,
               cmap=cmap)
    print("Evalutation complete")
    print(" ")
    print("________________________")
    print(" ")



if __name__ == "__main__":

    location = "Florida"

    # Extract Ground truth
    GT_path = f"/Users/tj/PycharmProjects/eval/semantic-segmentation/test_data/{location}_GT.png"
    GT = cv2.imread(GT_path)
    GT = np.array(GT)

    # Select one channel
    GT = GT[:, :, -1]

    # Generate a binary mask
    GT[GT > 0] = 1

    # Import/Extract MSI bands
    MSI_path = f"/Users/tj/PycharmProjects/Semantic-Segmentation_DeepLabV3/Water_Mask_Eval/Plots/{location}/{location}_test_site.npy"
    MSI = np.load(MSI_path)
    blue = crop(MSI[-1][:, :, 1], GT)
    green = crop(MSI[-1][:, :, 2], GT)
    red = crop(MSI[-1][:, :, 3], GT)
    NIR = crop(MSI[-1][:, :, 7], GT)
    SWIR1 = crop(MSI[-1][:, :, 10], GT)
    SWIR2 = crop(MSI[-1][:, :, 11], GT)



    NWDI_name = "NWDI"
    NWDI = (green - NIR)/(green + NIR)
    sns.distplot(NWDI, hist=False)
    plt.title(NWDI_name)
    plt.show()
    if location == "Florida":
        NWDI_thresh = 0
    elif location == "New_York":
        NWDI_thresh = 0
    elif location == "Shanghai":
        NWDI_thresh = 0
    else:
        print("No location")
    evaluate(data=NWDI, ground_truth=GT, thresh=NWDI_thresh, index_name=NWDI_name)



    MNDWI_name = "MNDWI"
    MNDWI = (green - SWIR2)/(green + SWIR2)
    sns.distplot(MNDWI, hist=False)
    plt.title(MNDWI_name)
    plt.show()
    if location == "Florida":
        MNDWI_thresh = 0
    elif location == "New_York":
        MNDWI_thresh = 0
    elif location == "Shanghai":
        MNDWI_thresh = 0
    else:
        print("No location")
    evaluate(data=MNDWI, ground_truth=GT, thresh=MNDWI_thresh, index_name=MNDWI_name)



    I_name = "I"
    I = ((green - NIR)/(green + NIR)) + ((blue - NIR)/(blue + NIR))
    sns.distplot(I, hist=False)
    plt.title(I_name)
    plt.show()
    if location == "Florida":
        I_thresh = 0
    elif location == "New_York":
        I_thresh = 0
    elif location == "Shanghai":
        I_thresh = 0
    else:
        print("No location")
    evaluate(data=I, ground_truth=GT, thresh=I_thresh, index_name=I_name)



    PI_name = "PI"
    PI = ((green - SWIR2)/(green + SWIR2)) + ((blue - NIR)/(blue + NIR))
    sns.distplot(PI, hist=False)
    plt.title(PI_name)
    plt.show()
    if location == "Florida":
        PI_thresh = 0
    elif location == "New_York":
        PI_thresh = 0
    elif location == "Shanghai":
        PI_thresh = 0.35
    else:
        print("No location")
    evaluate(data=PI, ground_truth=GT, thresh=PI_thresh, index_name=PI_name)



    AWEInsh_name = "AWEInsh"
    AWEInsh = (4 * (green - SWIR1) - (0.25 * NIR + 2.75 * SWIR2))/(green + NIR + SWIR1 + SWIR2)
    sns.distplot(AWEInsh, hist=False)
    plt.title(AWEInsh_name)
    plt.show()
    if location == "Florida":
        AWEInsh_thresh = 0
    elif location == "New_York":
        AWEInsh_thresh = 0
    elif location == "Shanghai":
        AWEInsh_thresh = 0
    else:
        print("No location")
    evaluate(data=AWEInsh, ground_truth=GT, thresh=AWEInsh_thresh, index_name=AWEInsh_name)



    AWEIsh_name = "AWEIsh"
    AWEIsh = (blue + 2.5 * green -1.5 * (NIR + SWIR1) - 0.25 * SWIR2)/(blue + green + NIR + SWIR1 + SWIR2)
    sns.distplot(AWEIsh, hist=False)
    plt.title(AWEIsh_name)
    plt.show()
    if location == "Florida":
        AWEIsh_thresh = 0
    elif location == "New_York":
        AWEIsh_thresh = 0
    elif location == "Shanghai":
        AWEIsh_thresh = 0
    else:
        print("No location")
    evaluate(data=AWEIsh, ground_truth=GT, thresh=AWEIsh_thresh, index_name=AWEIsh_name)




# PROPOSED WATER INDEX

    if location == "Florida":
        scalar = 0.03
    elif location == "New_York":
        scalar = 1.3
    elif location == "Shanghai":
        scalar = 1.37
    else:
        print("No location")


    # Proposed Water Index
    PWI_name = "PWI"

    i = scalar  # NDBI
    j = 1  # NWDI
    k = 1  # MNDWI
    PWI = (
    (i * ((SWIR2 - NIR) / (SWIR2 + NIR))) +
    (j * ((green - SWIR2) / (green + SWIR2))) +
    (k * ((green - NIR) / (green + NIR)))
    )
    sns.distplot(PWI, hist=False)
    plt.title(PWI_name)
    plt.show()
    if location == "Florida":
        PWI_thresh = 0
    elif location == "New_York":
        PWI_thresh = 0
    elif location == "Shanghai":
        PWI_thresh = 0
    else:
        print("No location")
    evaluate(data=PWI, ground_truth=GT, thresh=PWI_thresh, index_name=PWI_name)
