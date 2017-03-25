import numpy as np
import cv2

import matplotlib.pyplot as plt

class Lane:

    HIST_VALUES = 20

    def __init__(self):
        self.prev_start = None
        self.hist_left_poly = [[np.nan, np.nan, np.nan]]
        self.hist_right_poly = [[np.nan, np.nan, np.nan]]
        self.hist_curv_poly = [[np.nan, np.nan, np.nan]]
        self.left_poly = None
        self.right_poly = None


    def find_starting_point(self, bin_img):
        if self.prev_start is None:
            # We calculate midpoints based on color and on the complete
            # threshold.
            hist_col = np.sum(bin_img[bin_img.shape[0] / 2:, :], axis=0)
            midpoint_col = np.int(hist_col.shape[0] / 2)
            final_leftx = np.argmax(hist_col[:midpoint_col])
            final_rightx = np.argmax(hist_col[midpoint_col:]) + midpoint_col
            print(final_leftx, final_rightx)

            self.prev_start = [final_leftx, final_rightx]

        return self.prev_start

    def find_lanes(self, bin_img):
        start_lanes = self.find_starting_point(bin_img)

        out_img = np.dstack((bin_img, bin_img, bin_img)) * 255

        # Choose the number of sliding windows
        nwindows = 40
        # Set height of windows
        window_height = np.int(bin_img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = bin_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = start_lanes[0]
        rightx_current = start_lanes[1]
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 20
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = bin_img.shape[0] - (window + 1) * window_height
            win_y_high = bin_img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean
            # position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


        return [left_fit, right_fit, out_img]

    def find_smooth_lanes(self, bin_img):
        # Calculate the non smooth lanes first.
        left_fit, right_fit, hist_img = self.find_lanes(bin_img)

        # Verify whether we have too many values on the stack.
        fit_img = np.dstack((bin_img, bin_img, bin_img)) * 255

        # Verify that none of the values have a large percentage change, if this
        # is the case pop the oldest value if necessary and push the new one
        # on the stack.
        if np.all(abs((left_fit - self.hist_left_poly[-1]) /
                      self.hist_left_poly[-1]) <= [0.1, 0.1, 0.1]):
            if len(self.hist_left_poly) >= self.HIST_VALUES:
                self.hist_left_poly.pop(0)

            self.hist_left_poly.append(left_fit)
        # Now that we performed our first evaluation
        elif np.isnan(self.hist_left_poly[0][0]):
            self.hist_left_poly.pop(0)
            self.hist_left_poly.append(left_fit)

        left_fit = np.mean(np.stack(self.hist_left_poly), axis=0)

        if np.all(abs((right_fit - self.hist_right_poly[-1]) /
                      self.hist_right_poly[-1]) <= [0.1, 0.1, 0.1]):
            if len(self.hist_right_poly) >= self.HIST_VALUES:
                self.hist_right_poly.pop(0)

            self.hist_right_poly.append(right_fit)
        elif np.isnan(self.hist_right_poly[0][0]):
            self.hist_right_poly.pop(0)
            self.hist_right_poly.append(right_fit)

        right_fit = np.mean(self.hist_right_poly, axis=0)

        ploty = np.linspace(0, bin_img.shape[0]-1, bin_img.shape[0])
        left_fitx = (left_fit[0] * left_fit[0] * ploty +
                     left_fit[1] * ploty + left_fit[2])
        right_fitx = (right_fit[0] * right_fit[0] * ploty +
                      right_fit[1] * ploty + right_fit[2])

        self.left_poly = left_fit
        self.right_poly = right_fit

        # Plot the dots on the image
        #fit_img[np.int_(left_fitx), np.int_(ploty)] = [255, 0, 0]
        #fit_img[np.int_(right_fitx), np.int_(ploty)] = [0, 0, 255]

        return [left_fitx, right_fitx, ploty, fit_img, hist_img]

    def curv(self, left_fitx, right_fitx, ploty):
        if len(self.hist_curv_poly) > self.HIST_VALUES:
            self.hist_curv_poly.pop(0)

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 3600  # meters per pixel in x dimension
        # First invert ploty so that we can evaluate the average x and y at the
        # bottom
        ploty = ploty[::-1]

        offset = (right_fitx[max(ploty)] - left_fitx[max(ploty)]) * 3.7 / 700

        curv_fit = np.mean([self.left_poly, self.right_poly], axis=0)

        x_cr = np.array(ploty * curv_fit[0] * curv_fit[0] +
                        curv_fit[1] * ploty + curv_fit[2])
        fit_cr = np.polyfit(ploty * ym_per_pix, x_cr * xm_per_pix, 2)

        if np.all(abs((fit_cr - self.hist_curv_poly[-1]) /
                      self.hist_curv_poly[-1]) <= [0.1, 0.1, 0.1]):
            if len(self.hist_curv_poly) >= self.HIST_VALUES:
                self.hist_curv_poly.pop(0)

            self.hist_curv_poly.append(fit_cr)
        elif np.isnan(self.hist_curv_poly[0][0]):
            self.hist_curv_poly.pop(0)
            self.hist_curv_poly.append(curv_fit)

        fit_cr = np.mean(self.hist_curv_poly, axis=0)

        y_eval = max(ploty)
        curvad = (((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) /
                  np.absolute(2*fit_cr[0]))

        return [offset, curvad]

