import cv2


def perspective_correction_matrices(src_pnts, dst_pnts):
    road_camera_M = cv2.getPerspectiveTransform(src_pnts, dst_pnts)
    inv_road_camera_M = cv2.getPerspectiveTransform(dst_pnts, src_pnts)

    return [road_camera_M, inv_road_camera_M]


def perspective_correct(img, correction_matrix):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, correction_matrix, img_size,
                                 flags=cv2.INTER_CUBIC)
    return warped
