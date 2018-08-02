from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
from sklearn.preprocessing import normalize
import math

"""
1. Synthesize points on 3d soccer field model and visualize
2. generate virtual images from synthesized data and model 
3. save the synthesize data into mat file
"""

"""
function to generate random points.
"""


def generate_points():
    list_pts = []
    for i in range(200):
        xside = random.randint(0, 1)
        list_pts.append([xside * random.gauss(0, 5) + (1 - xside) * random.gauss(108, 5),
                         random.uniform(0, 70), random.uniform(0, 10)])

        list_pts.append([random.uniform(0, 108), random.gauss(63, 2), random.uniform(0, 10)])

    for i in range(0, 40):
        tmpx = random.gauss(54, 20)
        while tmpx > 108 or tmpx < 0:
            tmpx = random.gauss(54, 20)

        tmpy = random.gauss(32, 20)
        while tmpy > 63 or tmpy < 0:
            tmpy = random.gauss(32, 20)

        list_pts.append([tmpx, tmpy, random.uniform(0, 1)])

    pts_arr = np.array(list_pts, dtype=np.float32)

    return pts_arr


pts = generate_points()

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
compute the rays of these synthesized points
"""
proj_center = meta[0][0]["cc"][0]
rays = []

for i in range(0, len(pts)):
    relative = pts[i] - proj_center

    pan = math.atan(-relative[1] / relative[0])
    tilt = math.asin(relative[2] / np.linalg.norm(relative))
    rays.append([pan, tilt])

"""
This part is used to visualize the soccer model.
"""
fig = plt.figure(num=1, figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(0, 120)
ax.set_ylim(0, 70)
ax.set_zlim(0, 10)

for i in range(len(line_index)):
    x = [points[line_index[i][0]][0], points[line_index[i][1]][0]]
    y = [points[line_index[i][0]][1], points[line_index[i][1]][1]]
    z = [0, 0]
    ax.plot(x, y, z, color='g')

ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='r', marker='o')

plt.show()

"""
This part is used to show the synthesized images
"""
for i in range(annotation.size):

    img = np.zeros((720, 1280, 3), np.uint8)

    img.fill(255)
    camera = annotation[0][i]['camera'][0]

    k_paras = camera[0:3]
    k = np.array([[k_paras[2], 0, k_paras[0]], [0, k_paras[2], k_paras[1]], [0, 0, 1]])

    # rotation matrix
    rotation = np.zeros([3, 3])
    cv.Rodrigues(camera[3:6], rotation)

    # base position
    c = np.array(camera[6:9])
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

    # draw the feature points in images
    for j in range(len(pts)):
        p = np.array(pts[j])
        p = np.dot(k, np.dot(rotation, p - c))

        cv.circle(img, (int(p[0] / p[2]), int(p[1] / p[2])), color=(0, 0, 0), radius=8, thickness=2)

    cv.imshow("synthesized image", img)
    cv.waitKey(0)

"""
This part saves synthesized data to mat file
"""
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
