import cv2
import numpy as np


def perspective_correction_matrices(src_pnts, dst_pnts):
    road_camera_M = cv2.getPerspectiveTransform(src_pnts, dst_pnts)
    inv_road_camera_M = cv2.getPerspectiveTransform(dst_pnts, src_pnts)

    return [road_camera_M, inv_road_camera_M]


def perspective_correct(img, correction_matrix):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, correction_matrix, img_size,
                                 flags=cv2.INTER_CUBIC)
    return warped

def draw_polygon(img, ploty, left_fitx, right_fitx):
    # Create an output image to draw on and  visualize the result
    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    return color_warp


