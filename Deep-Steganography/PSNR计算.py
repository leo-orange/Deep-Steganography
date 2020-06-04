import cv2
import numpy as np
import math
def psnr1(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100

    return 10 * math.log10(255.0 ** 2 / mse)
gt = cv2.imread('./psnr/cover.png')
img = cv2.imread('./psnr/output.png')
print(psnr1(gt, img))
