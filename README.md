## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goal is to perform advanced lane finding. Initial research can be found in the [notebook](https://github.com/abossenbroek/CarND-Advanced-Lane-Lines/blob/master/Advanced_Lane_finding.ipynb).

After building the concept we continued with a module that performs all the tasks. The module
is `lanefinding` and is called in the program `detect_lanes.py`. We will discuss specifics of the
implementation below.

## Pipeline implementation
The pipeline implementation consist of a main program and a module that we specifically wrote
for this task. 

### Main program
The main program calculates the camera distortion correction matrix and the perspective matrix required
to change from image perspective to road perspective. It stores these matrices in a pickle for later reuse. 

Then it calls the pipeline for every frame in the picture. The pipeline consists of the following steps:
1. undistort the image for camera distortion
2. generate a road perspective of the image
3. isolate lanes by using Sobol thresholds as well as a yellow and white mask
4. generate points along two polygon that represent the left and right lanes
5. draw a polygon along the left and right lane
6. change the polygon from road perspective to original perspective
7. combine the polygon with the original image and add the road image as well as the information on the fitting
8. calculate the curvature and the offset of the car
9. add the curvature and the offset to the final image

All these steps leverage functions and a class in the `linefinding` module, which key aspect are explained below.

### LineFinding module
The `linefinding` module is built based on observations in the [notebook](https://github.com/abossenbroek/CarND-Advanced-Lane-Lines/blob/master/Advanced_Lane_finding.ipynb).
Some additional features were added. The first is the yellow and white mask identification. This was done by creating two files
with typical yellow and white points. These images were loaded in numpy and respectively converted to HSV and YUV color
space. The OpenCV function `cv2.inRange()` is used to create masks. These are combined with a bitwise and with the result
from the Sobol thresholding.

To reduce the impact of poor illuminated images the current implementation stacks the last 30 masks, averages them and
performs a range to remove noise. This is after many iterations of trying to reduce the sudden changes of the polynomials.
The first implementations used the mean of the last number of polynomials. This yielded in very poor results when we 
calculated the curvature and the offset. The curvature is calculated by taking the mean of the polynomials of the left and
right lane to remove potential noise.

## Possible improvements
The algorithm currently does not remove polynomials that are ill fitted. A possible improvement could involve filtering out 
poor fits. This could be done by inspecting the $R^2$ for example,  this is the residual of the fit. The lower the value the
less likely the polygon is a good fit.
