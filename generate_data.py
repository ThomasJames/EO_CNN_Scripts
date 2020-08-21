import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.image import imsave
import matplotlib.pyplot as plt
from skimage.transform import rescale
import skimage.color as color
import numpy as np
import PIL.Image as Image
import cv2
from PIL import Image, ImageOps
import random
import scipy

# Split the Images
def split_image(dim_pix, im, location, im_or_mask, folder, number):
    # Find the number of sub-Images that fit in rows
    rows = []
    for i in range((math.floor(im.shape[0] / dim_pix))):
        rows.append(i)
    # Find the number of sub-Images that fit in rows
    columns = []
    for i in range((math.floor(im.shape[1] / dim_pix))):
        columns.append(i)

    # Numerically identify the sub-Images
    a = 0
    for i in rows:
        for j in columns:
            # Check for 244 x 244 (Mask) or 244 x 244 x 3 (TC Images)
            if (im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
                  0 + dim_pix * i: dim_pix + (dim_pix * i)].shape[0]) == dim_pix:
                if (im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
                  0 + dim_pix * i: dim_pix + (dim_pix * i)].shape[1]) == dim_pix:

                    tile = im[0 + (dim_pix * j): dim_pix + (dim_pix * j),
                            0 + dim_pix * i: dim_pix + (dim_pix * i)]


                    # Stop white tiles for positive results
                    count = np.count_nonzero(tile == 1) == (dim_pix * dim_pix)
                    if count:
                        print(f"Tile {a} is only land")
                        all_black = np.tile(1, (dim_pix, dim_pix))
                        all_black[0][0] = 0
                        imsave(f"{folder}/{location}_{number}_{a}_{im_or_mask}.png",
                               all_black,
                               format="png",
                               cmap='Greys')
                    else:
                        # Save the 244 x 244 as an png file.
                        imsave(f"{folder}/{location}_{number}_{a}_{im_or_mask}.png",
                                tile,
                                format="png",
                                cmap='Greys')
                    a += 1
                else:
                    print("Out of shape")


# Salt and pepper
# Function by: Md. Rezwanul Haque (stolen from stack overflow)
def sp_noise(image, prob):
    '''
    Add salt and pepper noise to Images
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

# Blur

# Rotate 1

# Rotate 2


if __name__ == "__main__":

    """
    TC - Raw True colour Images
    GT - Ground Truth
    """

    location = "Shanghai"
    region = '12'

    # Load the raw Images
    TC = cv2.imread(f"Data/{location}_{region}/{location}_TC.png")
    TC = np.array(TC)

    print(TC.shape)

    plt.imshow(TC)
    plt.show()


    # Create a mask using the ground truth.
    # Convert the ground truth into a mask
    GT = cv2.imread(f"Data/{location}_{region}/{location}_GT.png")
    GT = np.array(GT)

    # Select one channel
    GT = GT[:, :, -1]

    # Generate a binary mask
    GT[GT > 0] = 1
    plt.imshow(GT, cmap="Blues")
    plt.show()


    TC_RAW = TC

    # plt.imsave(f"/Mask_Plots/GT_{region}_{location}.png", GT, cmap="tab20c")

    split_image(dim_pix=244, im=TC_RAW, location=location, im_or_mask=f"TC_RAW", number=f"Region_{region}",
                folder="Images")
    split_image(dim_pix=244, im=GT, location=location, im_or_mask=f"Mask_RAW", number=f"Region_{region}",
                folder="Masks")

    TC_noise = sp_noise(TC, 0.05)
    TC_Hflip = np.flip(TC, 1)
    GT_Hflip = np.flip(GT, 1)
    TC_Vflip = np.flip(TC, 0)
    GT_Vflip = np.flip(GT, 0)
    TC_Hflip_Vflip = np.flip(TC_Hflip, 0)
    GT_Hflip_Vflip = np.flip(GT_Hflip, 0)
    TC_Blur = cv2.medianBlur(TC, 5)

    split_image(dim_pix=244, im=TC_noise, location=location, im_or_mask=f"TC_noise", number=f"Region_{region}",
                folder="Images")
    split_image(dim_pix=244, im=GT, location=location, im_or_mask=f"Mask_noise", number=f"Region_{region}",
                folder="Masks")


    split_image(dim_pix=244, im=TC_Hflip_Vflip, location=location, im_or_mask=f"TC_Hflip_Vflip", number=f"Region_{region}",
                folder="Images")
    split_image(dim_pix=244, im=GT_Hflip_Vflip, location=location, im_or_mask=f"Mask_Hflip_Vflip", number=f"Region_{region}",
                folder="Masks")

    split_image(dim_pix=244, im=TC_Hflip, location=location, im_or_mask=f"TC_Hflip", number=f"Region_{region}",
                folder="Images")
    split_image(dim_pix=244, im=GT_Hflip, location=location, im_or_mask=f"Mask_Hflip", number=f"Region_{region}",
                folder="Masks")


    split_image(dim_pix=244, im=TC_Vflip, location=location, im_or_mask=f"TC_Vflip", number=f"Region_{region}",
                folder="Images")
    split_image(dim_pix=244, im=GT_Vflip, location=location, im_or_mask=f"Mask_Vflip", number=f"Region_{region}",
                folder="Masks")


    split_image(dim_pix=244, im=TC_Blur, location=location, im_or_mask=f"TC_blur", number=f"Region_{region}",
                folder="Images")
    split_image(dim_pix=244, im=GT, location=location, im_or_mask=f"Mask_blur", number=f"Region_{region}",
                folder="Masks")
