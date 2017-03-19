from LaneFinding.camera_calibration import camera_calibration


def main():
    print("Finding the calibration matrix.")
    mtx, dst = camera_calibration('camera_cal/calibration*.jpg')
    print("Found matrices to undistort the images from the camera.")


if __name__ == "__main__":
    main()
