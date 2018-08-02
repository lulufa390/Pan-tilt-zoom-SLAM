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
synthesize = sio.loadmat("./synthesize_data.mat")
pts = synthesize["pts"]
features = synthesize["features"]
rays = synthesize["rays"]

img = np.zeros((720, 1280, 3), np.uint8)

"""
initialize camera pose    
f, pan, tilt are 3 variables for camera pose
u, v are the image center
k is the intrinsic matrix 
"""
camera = annotation[0][0]['camera'][0]

u = camera[0]
v = camera[1]
f = camera[2]
k = np.array([[f, 0, u], [0, f, v], [0, 0, 1]])

"""
the rotation matrix
first get the base rotation
we will use radian instead of degree for pan and tilt angle
"""
base_rotation = np.zeros([3, 3])
cv.Rodrigues(meta[0][0]["base_rotation"][0], base_rotation)

pan = annotation[0][0]['ptz'].squeeze()[0] * math.pi / 180
tilt = annotation[0][0]['ptz'].squeeze()[1] * math.pi / 180

rotation = np.dot(np.array([[1, 0, 0], [0, math.cos(tilt), math.sin(tilt)], [0, -math.sin(tilt), math.cos(tilt)]]),
                  np.array([[math.cos(pan), 0, -math.sin(pan)], [0, 1, 0], [math.sin(pan), 0, math.cos(pan)]]))
rotation = np.dot(rotation, base_rotation)

"""
projection center of ptz camera
"""
c = np.array(camera[6:9])

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
print(p.shape)

for i in range(1, annotation.size):

    """
    prediction
    """
    tmp_x = x[i - 1] + [delta_pan, delta_tilt, delta_zoom]
    tmp_p = p[i - 1] + 0.1 * np.ones([3, 3])
    # print("+====================+")
    # print(p)
    """
    clear the frame
    """
    img.fill(255)

    """
    get the next observation of feature points for next frame (using ground truth camera pose)
    """
    tmp_camera = annotation[0][i]['camera'][0]

    tmp_paras = tmp_camera[0:3]
    tmp_k = np.array([[tmp_paras[2], 0, tmp_paras[0]], [0, tmp_paras[2], tmp_paras[1]], [0, 0, 1]])

    tmp_pt = annotation[0][i]['ptz'].squeeze()
    tmp_pt = tmp_pt * math.pi / 180
    tmp_rotation = np.dot(
        np.array(
            [[1, 0, 0], [0, math.cos(tmp_pt[1]), math.sin(tmp_pt[1])], [0, -math.sin(tmp_pt[1]), math.cos(tmp_pt[1])]]),
        np.array(
            [[math.cos(tmp_pt[0]), 0, -math.sin(tmp_pt[0])], [0, 1, 0], [math.sin(tmp_pt[0]), 0, math.cos(tmp_pt[0])]]))
    tmp_rotation = np.dot(tmp_rotation, base_rotation)

    features = np.ndarray([len(pts), 2])

    """
    draw the feature points in images
    """

    for j in range(len(pts)):
        pos = np.array(pts[j])
        pos = np.dot(tmp_k, np.dot(tmp_rotation, pos - c))
        features[j] = [int(pos[0] / pos[2]), int(pos[1] / pos[2])]

        # cv.circle(img, (int(p[0] / p[2]), int(p[1] / p[2])), color=(0, 0, 0), radius=8, thickness=2)


    # print(features.shape)

    """
    yk = zk - h(xk|k-1)
    """
    hx = np.ndarray([len(pts), 2])

    print(u, v)
    print(tmp_x)
    for j in range(len(pts)):
        hx[j][0] = f * math.tan(rays[j][0] - tmp_x[0]) + u
        hx[j][1] = f * math.tan(rays[j][1] - tmp_x[1]) + v
        cv.circle(img, (int(hx[j][0]), int(hx[j][1])), color=(0, 0, 0), radius=8, thickness=2)

        print("fuck")
        print(features[j])
        print(hx[j])

    # print(hx)

    cv.imshow("synthesized image", img)

    cv.waitKey(0)

    """
    get the next pose Pn_init
    """

    # f += delta_zoom
    # pan += delta_pan
    # tilt += delta_tilt

    """
    Update Pn_init with the observation Z1.
    """
