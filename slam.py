"""
PTZ camera SLAM tested on synthesized data  2018.8
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
from sklearn.preprocessing import normalize
import math
import synthesize

"""                       
load the soccer field model
"""
soccer_model = sio.loadmat("./two_point_calib_dataset/util/highlights_soccer_model.mat")
line_index = soccer_model['line_segment_index']
points = soccer_model['points']

"""
load the sequence annotation
"""
seq = sio.loadmat("./two_point_calib_dataset/highlights/seq3_anno.mat")
annotation = seq["annotation"]
meta = seq['meta']

"""
load the synthesized data
"""
data = sio.loadmat("./synthesize_data.mat")
pts = data["pts"]
features = data["features"]
rays = data["rays"]

img = np.zeros((720, 1280, 3), np.uint8)

"""
initialize camera pose    
f, pan, tilt are 3 variables for camera pose
u, v are the image center
k is the intrinsic matrix 
"""
u, v, f = annotation[0][0]['camera'][0][0:3]

"""
the rotation matrix         
first get the base rotation
we will use radian instead of degree for pan and tilt angle
"""
base_rotation = np.zeros([3, 3])
cv.Rodrigues(meta[0][0]["base_rotation"][0], base_rotation)

pan, tilt, _ = annotation[0][0]['ptz'].squeeze() * math.pi / 180

"""
projection center of ptz camera
"""
c = meta[0][0]["cc"][0]

delta_zoom = 0
delta_pan = 0
delta_tilt = 0

"""
initialize the camera pose 
"""
x = np.ndarray([annotation.size, 3])
x[0] = [pan, tilt, f]

"""
initialize the covariance matrix
"""
p = np.ndarray([annotation.size, 3, 3])
tmp = np.diag([0.01, 0.01, 0.01])
p[0] = tmp


def update(previous_x, previous_p, observe):
    hx = np.ndarray([len(pts), 2])

    for j in range(len(pts)):
        hx[j][0] = f * math.tan(rays[j][0] - previous_x[0]) + u
        hx[j][1] = - f * math.tan(rays[j][1] - previous_x[1]) + v
        # cv.circle(img, (int(hx[j][0]), int(hx[j][1])), color=(0, 0, 0), radius=8, thickness=2)

        print("fuck")
        print(features[j])
        print(hx[j])

    y = observe.flatten() - hx.flatten()

    """
    Sk = Hk*Pk|k-1*Hk^T + Rk
    """

    jacobi_h = np.ndarray([2 * len(pts), 3])

    for j in range(len(pts)):
        jacobi_h[2 * j][0] = -f / math.pow(math.cos(rays[j][0] - previous_x[0]), 2)
        jacobi_h[2 * j][1] = 0
        jacobi_h[2 * j][2] = math.tan(rays[j][0] - previous_x[0])
        jacobi_h[2 * j + 1][0] = 0
        jacobi_h[2 * j + 1][1] = -f / math.pow(math.cos(rays[j][1] - previous_x[1]), 2)
        jacobi_h[2 * j + 1][2] = math.tan(rays[j][1] - previous_x[1])

    s = np.dot(np.dot(jacobi_h, previous_p), jacobi_h.T) + 0.01 * np.ones([2 * len(pts), 2 * len(pts)])

    k = np.dot(np.dot(previous_p, jacobi_h.T), np.linalg.inv(s))

    return [np.transpose(previous_x + np.dot(k, y)), np.dot((np.eye(3) - np.dot(k, jacobi_h)), tmp_p)]


"""
the main loop for EKF algorithm
i is the index for next state, as we the first camera pose is already known, i begins from 1. 
"""
for i in range(1, annotation.size):

    """
    prediction
    """
    tmp_x = x[i - 1] + [delta_pan, delta_tilt, delta_zoom]
    tmp_p = p[i - 1] + 0.01 * np.ones([3, 3])

    """
    clear the frame
    """
    img.fill(255)

    """
    get the next observation of feature points for next frame (using ground truth camera pose)
    """
    tmp_f = annotation[0][i]['camera'][0][2]
    pan, tilt, _ = annotation[0][i]['ptz'].squeeze() * math.pi / 180

    features = np.ndarray([len(pts), 2])

    for j in range(len(pts)):
        pos = np.array(pts[j])
        features[j] = synthesize.from_3d_to_2d(u, v, tmp_f, pan, tilt, c, base_rotation, pos)
        cv.circle(img, (int(features[j][0]), int(features[j][1])), color=(0, 0, 0), radius=8, thickness=2)

    """
    yk = zk - h(xk|k-1)
    """

    x[i], p[i] = update(tmp_x, tmp_p, features)

    # x[i] = np.transpose(tmp_x + np.dot(k, y))
    #
    # p[i] = np.dot((np.eye(3) - np.dot(k, jacobi_h)), tmp_p)

    delta_pan, delta_tilt, delta_zoom = x[i] - x[i - 1]



    cv.imshow("synthesized image", img)

    cv.waitKey(0)

print(x)

