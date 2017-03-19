import cv2
import numpy as np


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        dir_sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    elif orient == 'y':
        dir_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    abs_sobel = np.absolute(dir_sobel)
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Use OpenCV inRange function to build a mask.
    binary_output = cv2.inRange(scaled_sobel, thresh[0], thresh[1])

    # Return the result
    return [binary_output, dir_sobel]


def combined_sobel_thresh(img, abs_kernel=3, abs_thresh=(0, 255),
                 mag_kernel=3, mag_thresh=(0, 255),
                 dir_kernel=3, dir_thresh=(0, np.pi/2)):
    x_sobel_mask, x_sobel = abs_sobel_thresh(img, 'x', sobel_kernel=abs_kernel,
                                             thresh=abs_thresh)
    y_sobel_mask, y_sobel = abs_sobel_thresh(img, 'y', sobel_kernel=abs_kernel,
                                             thresh=abs_thresh)
    # Try to recycle the the sobel values, which is only possible if we have the
    # same kernel.
    if abs_kernel != mag_kernel:
        x_sobel_mask, x_sobel = abs_sobel_thresh(img, 'x',
                                                 sobel_kernel=mag_kernel,
                                                 thresh=abs_thresh)
        y_sobel_mask, y_sobel = abs_sobel_thresh(img, 'y',
                                                 sobel_kernel=mag_kernel,
                                                 thresh=abs_thresh)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(x_sobel * x_sobel + y_sobel * y_sobel)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Calculate the gradient magnitude mask.
    gradmag_mask = cv2.inRange(gradmag, mag_thresh[0], mag_thresh[1])

    # Try to recycle the the sobel values, which is only possible if we have the
    # same kernel.
    if dir_kernel != mag_kernel:
        x_sobel_mask, x_sobel = abs_sobel_thresh(img, 'x',
                                                 sobel_kernel=dir_kernel,
                                                 thresh=dir_thresh)
        y_sobel_mask, y_sobel = abs_sobel_thresh(img, 'y',
                                                 sobel_kernel=dir_kernel,
                                                 thresh=dir_thresh)

    dir = np.arctan2(np.absolute(y_sobel), np.absolute(x_sobel))
    dir_mask = cv2.inRange(dir, dir_thresh[0], dir_thresh[1])

    final_mask = cv2.bitwise_or(cv2.bitwise_and(x_sobel_mask, y_sobel_mask),
                                cv2.bitwise_and(gradmag_mask, dir_mask))

    return final_mask

def white_thresh(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lower_white = np.array([0, 0, 120], dtype=np.uint8)
    upper_white = np.array([360, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hls, lower_white, upper_white)
    return mask

def yellow_thresh(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([138,140,30], dtype=np.uint8)
    upper_yellow = np.array([190, 180, 93], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask
