"""
1. Synthesize points on 3d soccer field model and visualize
2. generate virtual images from synthesized data and model
3. save the synthesize data into mat file
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
from sklearn.preprocessing import normalize
from math import *

"""
function to generate random points.
'num' is the number of random points to generate
"""


def generate_points(num):
    list_pts = []
    random.seed(1)

    for i in range(num):
        choice = random.randint(0, 6)
        if choice < 3:
            xside = random.randint(0, 1)
            list_pts.append([xside * random.gauss(0, 5) + (1 - xside) * random.gauss(108, 5),
                             random.uniform(0, 70), random.uniform(0, 10)])
        elif choice < 5:
            list_pts.append([random.uniform(0, 108), random.gauss(63, 2), random.uniform(0, 10)])
        else:
            tmpx = random.gauss(54, 20)
            while tmpx > 108 or tmpx < 0:
                tmpx = random.gauss(54, 20)

            tmpy = random.gauss(32, 20)
            while tmpy > 63 or tmpy < 0:
                tmpy = random.gauss(32, 20)

            list_pts.append([tmpx, tmpy, random.uniform(0, 1)])

    pts_arr = np.array(list_pts, dtype=np.float32)
    return pts_arr


"""
function from 3d -> 2d
u, v, f, pan, tilt, c, base_r is the parameters of camera
pos is the 3d position of that feature points
"""


def from_3d_to_2d(u, v, f, pan, tilt, c, base_r, pos):
    # camera = annotation['camera'][0]
    # u = camera[0]
    # v = camera[1]
    # f = camera[2]
    k = np.array([[f, 0, u], [0, f, v], [0, 0, 1]])

    # pan = annotation['ptz'].squeeze()[0] * math.pi / 180
    # tilt = annotation['ptz'].squeeze()[1] * math.pi / 180

    rotation = np.dot(np.array([[1, 0, 0], [0, cos(tilt), sin(tilt)], [0, -sin(tilt), cos(tilt)]]),
                      np.array([[cos(pan), 0, -sin(pan)], [0, 1, 0], [sin(pan), 0, cos(pan)]]))
    rotation = np.dot(rotation, base_r)

    # c = np.array(camera[6:9])
    pos = np.dot(k, np.dot(rotation, pos - c))

    return [pos[0] / pos[2], pos[1] / pos[2]]


"""
function from pan/tilt to 2d
u, v, f, camera_pan, camera_tilt is the parameters of camera pose
pan and tilt is the angle for that feature point
"""


def compute_pose_relative_angles(pan, tilt, camera_pan, camera_tilt):
    test_pan = atan((tan(pan) * cos(camera_pan) - sin(camera_pan)) \
                    / (tan(pan) * sin(camera_pan) * cos(camera_tilt) + tan(tilt) * sqrt(tan(pan) * tan(pan) + 1) * sin(
        camera_tilt) + cos(camera_tilt) * cos(camera_pan)))

    test_tilt = atan(-(tan(pan) * sin(camera_tilt) * sin(camera_pan) - tan(tilt) * sqrt(tan(pan) * tan(pan) + 1) * cos(
        camera_tilt) + sin(camera_tilt) * cos(camera_pan)) \
                     / sqrt(pow(tan(pan) * cos(camera_pan) - sin(camera_pan), 2) + pow(
        tan(pan) * sin(camera_pan) * cos(camera_tilt) + tan(tilt) * sqrt(tan(pan) * tan(pan) + 1) * sin(
            camera_tilt) + cos(camera_tilt) * cos(camera_pan), 2)))

    return test_pan, test_tilt


def from_pan_tilt_to_2d(u, v, f, camera_pan, camera_tilt, pan, tilt):

    test_pan, test_tilt = compute_pose_relative_angles(pan, tilt, camera_pan, camera_tilt)

    # x = f * math.tan(pan - camera_pan) + u
    dx = f * tan(test_pan)
    x = dx + u
    # y = -f * tan(tilt - camera_tilt) + v
    y = -sqrt(f * f + dx * dx) * tan(test_tilt) + v

    return [x, y]


"""
function to compute rays from 3D points
proj_center is camera center, and base_r is the base rotation matrix
"""


def compute_rays(proj_center, pos, base_r):
    relative = np.dot(base_r, np.transpose(pos - proj_center))
    x, y, z = relative

    pan = atan(x / z)
    tilt = atan(-y / sqrt(x * x + z * z))
    return [pan, tilt]


"""
function to save the synthesized data to .mat file
And randomly generate the 16-dimension features
"""


def save_to_mat_radian(pts, rays):
    key_points = dict()
    features = []

    # generate features randomly
    for i in range(len(pts)):
        vec = np.random.random(16)
        vec = vec.reshape(1, 16)
        vec = normalize(vec, norm='l2')
        vec = np.squeeze(vec)
        features.append(vec)

    key_points['features'] = features
    key_points['pts'] = pts
    key_points['rays'] = rays

    sio.savemat('synthesize_data.mat', mdict=key_points)


def save_to_mat_degree(pts, rays):
    key_points = dict()
    features = []

    # generate features randomly
    for i in range(len(pts)):
        vec = np.random.random(16)
        vec = vec.reshape(1, 16)
        vec = normalize(vec, norm='l2')
        vec = np.squeeze(vec)
        features.append(vec)

    key_points['features'] = features
    key_points['pts'] = pts
    key_points['rays'] = np.asarray(rays) * (180 / pi)

    sio.savemat('synthesize_data.mat', mdict=key_points)


"""
this function is to draw the 3D model of soccer field
"""


def draw_3d_model(line_index, line_points, features):
    plt.ion()
    fig = plt.figure(num=1, figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 70)
    ax.set_zlim(0, 10)
    for i in range(len(line_index)):
        x = [line_points[line_index[i][0]][0], line_points[line_index[i][1]][0]]
        y = [line_points[line_index[i][0]][1], line_points[line_index[i][1]][1]]
        z = [0, 0]
        ax.plot(x, y, z, color='g')

    ax.scatter(features[:, 0], features[:, 1], features[:, 2], color='r', marker='o')
    plt.show()


"""
this function draws the lines of soccer field in one image from a specific camera pose
"""


def draw_soccer_line(img, u, v, f, pan, tilt, base_r, c, line_index, points):
    k = np.array([[f, 0, u], [0, f, v], [0, 0, 1]])

    rotation = np.dot(np.array([[1, 0, 0], [0, cos(tilt), sin(tilt)], [0, -sin(tilt), cos(tilt)]]),
                      np.array([[cos(pan), 0, -sin(pan)], [0, 1, 0], [sin(pan), 0, cos(pan)]]))
    rotation = np.dot(rotation, base_r)

    image_points = np.ndarray([len(points), 2])

    # get points and draw lines
    for j in range(len(points)):
        p = np.array([points[j][0], points[j][1], 0])
        p = np.dot(k, np.dot(rotation, p - c))
        image_points[j][0] = p[0] / p[2]
        image_points[j][1] = p[1] / p[2]

    for j in range(len(line_index)):
        begin = line_index[j][0]
        end = line_index[j][1]
        cv.line(img, (int(image_points[begin][0]), int(image_points[begin][1])),
                (int(image_points[end][0]), int(image_points[end][1])), (0, 0, 255), 5)
