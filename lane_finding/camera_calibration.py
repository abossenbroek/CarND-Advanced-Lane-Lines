import numpy as np
import cv2
import glob


def find_corners(filename, idx, num_corners, objpoints, imgpoints):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, num_corners, None)

    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

    return [objpoints, imgpoints]


def camera_calibration(calibration_images):
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    img_size = None
    # Make a list of calibration images
    images = glob.glob(calibration_images)
    last_fn = None

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        objpoints, imgpoints = find_corners(fname, idx, (9, 6),
                                            objpoints, imgpoints)
        last_fn = fname


    img = cv2.imread(last_fn)
    img_size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       img_size, None, None)

    return [mtx, dist]
