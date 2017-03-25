import lanefinding as lf
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import os.path
import pickle

DISTORT_MATRIX_FN = "distort_matrix.p"

mtx = None
road_M = None
inv_road_M = None
lane = lf.Lane()

def find_lane(img):
    global mtx
    global road_M
    global inv_road_M
    global lane

    # Let us first correct the image distortion caused by the camera.
    undist_img = lf.perspective_correct(img, mtx)
    persp_corr_img = lf.perspective_correct(undist_img, road_M)
    bin_img, white_msk, yellow_msk = lf.isolate_lanes(persp_corr_img)
    left_fitx, right_fitx, ploty, fit_img = lane.find_smooth_lanes(bin_img, yellow_msk, white_msk)
    polygon = lf.draw_polygon(persp_corr_img, ploty, left_fitx, right_fitx)
    # Change the polygon with detected lanes back to our original road perspective.
    polygon = lf.perspective_correct(polygon, inv_road_M)
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, polygon, 0.3, 0)

    #TODO: add curv and offset in dashboard
    curv, offset = lane.curv(left_fitx, right_fitx, ploty)

    return result


def main():
    global mtx
    global road_M
    global inv_road_M

    if os.path.isfile(DISTORT_MATRIX_FN):
        print("Read perspective matrices from file")
        with open(DISTORT_MATRIX_FN, "rb") as f:
            mtx, dst, road_M, inv_road_M = pickle.load(f)
    else:
        print("Finding the calibration matrix.")
        mtx, dst = lf.camera_calibration('camera_cal/calibration*.jpg')
        print("Found matrices to undist the images from the camera.")
        print("Next let us find the matrix that will allow us to perform perspective correction.")
        src_pnts = np.float32([[1033, 670],
                               [275, 670],
                               [705, 460],
                               [580, 460]])

        dst_pnts = np.float32([[1000, 700],
                               [200, 700],
                               [1000, 0],
                               [200, 0]])
        road_M, inv_road_M = lf.perspective_correction_matrices(src_pnts, dst_pnts)
        print("Found perspective correction matrices")
        print("Write perspective matrices to file")
        with open(DISTORT_MATRIX_FN, "wb") as f:
            pickle.dump([mtx, dst, road_M, inv_road_M], f)


    print("About to process movie")

    white_output = 'project_video_out_five.mp4'
    clip1 = VideoFileClip("project_video.mp4").subclip(38, 45)
    white_clip = clip1.fl_image(find_lane)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


if __name__ == "__main__":
    main()
