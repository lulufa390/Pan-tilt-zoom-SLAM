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
1. load the soccer field model
2. load the sequence annotation
3. load the synthesized data
"""
soccer_model = sio.loadmat("./two_point_calib_dataset/util/highlights_soccer_model.mat")
line_index = soccer_model['line_segment_index']
points = soccer_model['points']

seq = sio.loadmat("./two_point_calib_dataset/highlights/seq3_anno.mat")
annotation = seq["annotation"]
meta = seq['meta']

data = sio.loadmat("./synthesize_data.mat")
pts = data["pts"]
features = data["features"]
rays = data["rays"]

img = np.zeros((720, 1280, 3), np.uint8)

"""
get the first camera pose 
f, pan, tilt are 3 variables for camera pose
u, v are the image center
"""
u, v, f = annotation[0][0]['camera'][0][0:3]
pan, tilt, _ = annotation[0][0]['ptz'].squeeze() * math.pi / 180

"""
base_rotation: the base rotation matrix of PTZ camera         
c: projection center of ptz camera
"""
base_rotation = np.zeros([3, 3])
cv.Rodrigues(meta[0][0]["base_rotation"][0], base_rotation)
c = meta[0][0]["cc"][0]

"""
1. initialize the camera pose array 
2. initialize the covariance matrix array 
3. speed model for PTZ camera
"""
x = np.ndarray([annotation.size, 3])
x[0] = [pan, tilt, f]

p = np.ndarray([annotation.size, 3, 3])
tmp = np.diag([0.01, 0.01, 0.01])
p[0] = tmp

delta_pan, delta_tilt, delta_zoom = [0 for i in range(3)]


def compute_jacobi(theta, phi, foc, ray):
    jacobi_h = np.ndarray([2, 3])

    jacobi_h[0][0] = -foc / math.pow(math.cos(ray[0] - theta), 2)
    jacobi_h[0][1] = 0
    jacobi_h[0][2] = math.tan(ray[0] - theta)
    jacobi_h[1][0] = 0
    jacobi_h[1][1] = foc / math.pow(math.cos(ray[1] - phi), 2)
    jacobi_h[1][2] = -math.tan(ray[1] - phi)

    return jacobi_h

# print( compute_jacobi(0.9, 0.1, 3500, [0.9,0.1]) - compute_jacobi(0.9, 0.1, 3500, [0.9001,0.1001]))


def update(previous_x, previous_p, observe, u, v, rays):
    hx = np.ndarray([len(pts), 2])

    for j in range(len(rays)):
        hx[j][0] = previous_x[2] * math.tan(rays[j][0] - previous_x[0]) + u
        hx[j][1] = - previous_x[2] * math.tan(rays[j][1] - previous_x[1]) + v
        cv.circle(img, (int(hx[j][0]), int(hx[j][1])), color=(255, 0, 0), radius=8, thickness=2)

    y = []
    jacobis = []
    cnt = 0
    for j in range(len(rays)):
        if 0 < observe[j][0] < 1280 and 0 < observe[j][1] < 720 and 0 < hx[j][0] < 1280 and 0 < hx[j][1] < 720:
            y.append(observe[j][0] - hx[j][0])
            y.append(observe[j][1] - hx[j][1])
            jacobis.append(compute_jacobi(previous_x[0], previous_x[1], previous_x[2], rays[j]))
            cnt += 1

    y = np.array(y)
    jacobis = np.array(jacobis)
    jacobis = jacobis.reshape((-1, 3))

    """
    Sk = Hk*Pk|k-1*Hk^T + Rk
    """

    s = np.dot(np.dot(jacobis, previous_p), jacobis.T) + 0.01 * np.ones([2 * cnt, 2 * cnt])

    k = np.dot(np.dot(previous_p, jacobis.T), np.linalg.inv(s))

    updated_x = np.transpose(previous_x + np.dot(k, y))
    updated_p = np.dot((np.eye(3) - np.dot(k, jacobis)), tmp_p)
    return [updated_x, updated_p]


"""
the main loop for EKF algorithm
i is the index for next state, as we the first camera pose is already known, i begins from 1. 
"""

for i in range(1, 2):

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

    x[i], p[i] = update(tmp_x, tmp_p, features, u, v, rays)

    delta_pan, delta_tilt, delta_zoom = x[i] - x[i - 1]

    cv.imshow("synthesized image", img)

    cv.waitKey(0)

print(x)
