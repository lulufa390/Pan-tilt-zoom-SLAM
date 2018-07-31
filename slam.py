from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
from sklearn.preprocessing import normalize


"""
load the soccer field model
"""
soccer_model = sio.loadmat("./two_point_calib_dataset/util/highlights_soccer_model.mat")
line_index = soccer_model['line_segment_index']
points = soccer_model['points']


"""
# load the sequence annotation
"""
seq = sio.loadmat("./two_point_calib_dataset/highlights/seq3_anno.mat")
annotation = seq["annotation"]
meta = seq['meta']

"""
load the synthesized data
"""

synthesize = sio.loadmat("./synthesize_data.mat")
pts = synthesize["pts"]


img = np.zeros((720, 1280, 3), np.uint8)
#
#
# img.fill(255)


"""
initialize camera pose
"""
camera = annotation[0][0]['camera'][0]

k_paras = camera[0:3]
f = k_paras[2]
k = np.array([[f, 0, k_paras[0]], [0, f, k_paras[1]], [0, 0, 1]])

# rotation matrix
angles = camera[3:6]
pt = camera[7:9]
rotation = np.zeros([3, 3])
cv.Rodrigues(angles, rotation)

# base position
c = np.array(camera[6:9])
image_points = np.ndarray([len(points), 2])

delta_zoom = 0
delta_pan = 0
delta_tilt = 0


for i in range(1, annotation.size):

    #clear the image
    img.fill(255)

    # f += 10
    # k = np.array([[f, 0, k_paras[0]], [0, f, k_paras[1]], [0, 0, 1]])

    # angles += 0.001
    # print(angles)
    # cv.Rodrigues(angles, rotation)
    # get points and draw lines

    """
    get the next observation of ground truth camera pose for next frame
    """
    tmp_camera = annotation[0][i]['camera'][0]

    tmp_paras = tmp_camera[0:3]
    tmp_k = np.array([[tmp_paras[2], 0, tmp_paras[0]], [0, tmp_paras[2], tmp_paras[1]], [0, 0, 1]])

    # rotation matrix
    tmp_rotation = np.zeros([3, 3])
    cv.Rodrigues(tmp_camera[3:6], tmp_rotation)


    features = []
    # draw the feature points in images
    for j in range(len(pts)):
        p = np.array(pts[j])
        p = np.dot(tmp_k, np.dot(tmp_rotation, p - c))
        features.append(p)
        cv.circle(img, (int(p[0] / p[2]), int(p[1] / p[2])), color=(0, 0, 0), radius=8, thickness=2)

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
