"""
1. class 'TransFuntion' provides functions for projection and ray computing.

2. All the input and output angles of functions in class 'TransFunction' is in degree!!

3. The type of variables is mostly np.float64(default), but it does not have much influence if you use np.float32.

Created by Luke, 2018.9
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
from sklearn.preprocessing import normalize
from math import *
from mpl_toolkits.mplot3d import Axes3D

# @todo this class is actually the projection and back-projection in 3D vision
# for example: from_3d_to_2d() is projection and from from_2d_to_3d() is back-projection
# we cane put these function into two classes Camera and PTZCamera
# Camera: is the general perspective camera
# PTZCamera: is pan-tilt-zoom camera
# if we use base_rotation, the general camera becomes pan-tilt-zoom camera
class TransFunction:
    @staticmethod
    def from_3d_to_2d(u, v, f, p, t, c, base_r, pos):
        """
        project 3d point in world coordinate to image
        :param u: camera parameter u
        :param v: camera parameter v
        :param f: camera parameter f
        :param p: pan
        :param t: tilt
        :param c: projection center: array [3]
        :param base_r: base rotation matrix: array [3, 3]
        :param pos: 3d point (world coordinate!): array [3]
        :return: tuple (x, y) in image
        """
        pan = radians(p)
        tilt = radians(t)

        k = np.array([[f, 0, u],
                      [0, f, v],
                      [0, 0, 1]])

        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),

                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))

        rotation = np.dot(rotation, base_r)

        position = np.dot(k, np.dot(rotation, pos - c))

        return position[0] / position[2], position[1] / position[2]

    @staticmethod
    def from_2d_to_3d(u, v, f, p, t, c, base_r, point2d):
        """
        this function backproject image to 3d world coordinate
        z can be set to different value
        :param u: camera parameter
        :param v: camera parameter
        :param f: camera f
        :param p: camera p
        :param t: camera t
        :param c: camera projection center array [3]
        :param base_r: base rotation array [3, 3]
        :param point2d: point on image
        :return: 3d world coordinate point with z = 1
        """

        z = 0
        pan = radians(p)
        tilt = radians(t)

        k = np.array([[f, 0, u],
                      [0, f, v],
                      [0, 0, 1]])

        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),

                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))

        rotation = np.dot(rotation, base_r)

        inv_mat = np.linalg.inv(np.dot(k, rotation))

        coe = (z - c[2]) / (inv_mat[2, 0] * point2d[0] + inv_mat[2, 1] * point2d[1] + inv_mat[2, 2])

        p = np.dot(inv_mat, coe * np.array([point2d[0], point2d[1], 1])) + c

        return p

    @staticmethod
    def from_pan_tilt_to_2d(u, v, f, c_p, c_t, p, t):
        """
        project ray to image
        :param u: camera parameter u
        :param v: camera parameter v
        :param f: camera parameter f
        :param c_p: camera pan
        :param c_t: camera tilt
        :param p: ray theta
        :param t: ray phi
        :return: tuple (x, y) in image
        """
        pan = radians(p)
        tilt = radians(t)
        camera_pan = radians(c_p)
        camera_tilt = radians(c_t)

        #@todo how to get these equations?
        #@todo add these equations in the document
        relative_pan = atan((tan(pan) * cos(camera_pan) - sin(camera_pan)) /
                            (tan(pan) * sin(camera_pan) * cos(camera_tilt) +
                             tan(tilt) * sqrt(tan(pan) * tan(pan) + 1) *
                             sin(camera_tilt) + cos(camera_tilt) * cos(camera_pan)))

        relative_tilt = atan(-(tan(pan) * sin(camera_tilt) * sin(camera_pan) -
                               tan(tilt) * sqrt(tan(pan) * tan(pan) + 1) *
                               cos(camera_tilt) + sin(camera_tilt) * cos(camera_pan)) /
                             sqrt(pow(tan(pan) * cos(camera_pan) - sin(camera_pan), 2) +
                                  pow(tan(pan) * sin(camera_pan) * cos(camera_tilt) +
                                      tan(tilt) * sqrt(tan(pan) * tan(pan) + 1) *
                                      sin(camera_tilt) + cos(camera_tilt) * cos(camera_pan), 2)))

        dx = f * tan(relative_pan)
        x = dx + u
        y = -sqrt(f * f + dx * dx) * tan(relative_tilt) + v
        return x, y

    @staticmethod
    def from_2d_to_pan_tilt(u, v, f, c_p, c_t, x, y):
        """
        from image to ray
        :param u: camera parameter u
        :param v: camera parameter v
        :param f: camera parameter f
        :param c_p: camera pan
        :param c_t: camera tilt
        :param x: 2d image point x
        :param y: 2d image point y
        :return: ray tuple (theta, phi) in degree
        """
        pan = radians(c_p)
        tilt = radians(c_t)

        # @todo what are thse xxx_skim?
        theta_skim = atan((x - u) / f)
        phi_skim = atan((y - v) / (-f * sqrt(1 + pow((x - u) / f, 2))))

        x3d_skim = tan(theta_skim)
        y3d_skim = -tan(phi_skim) * sqrt(pow(tan(theta_skim), 2) + 1)

        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),

                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))

        rotation = np.linalg.inv(rotation)

        x3d, y3d, z3d = np.dot(rotation, np.array([x3d_skim, y3d_skim, 1]))

        theta = atan(x3d / z3d)
        phi = atan(-y3d / sqrt(x3d * x3d + z3d * z3d))

        return degrees(theta), degrees(phi)

    @staticmethod
    def compute_rays(proj_center, pos, base_r):
        """
        from 3d point (world coordinate) to ray
        :param proj_center: projection center: array [3]
        :param pos: 3-d position: 3d point in world coordinate: array [3]
        :param base_r: base rotation: array [3, 3]
        :return: ray tuple (theta, phi) in degree
        """
        relative = np.dot(base_r, np.transpose(pos - proj_center))
        x, y, z = relative
        theta = atan(x / z)
        phi = atan(-y / sqrt(x * x + z * z))

        return degrees(theta), degrees(phi)

    @staticmethod
    def from_ray_to_relative_3d(t, p):
        """
        from ray to 3d camera coordinate
        :param t: ray theta angle
        :param p: ray phi angle
        :return: 3d point in camera coordinate in tuple (x, y, 1)
        """
        theta = radians(t)
        phi = radians(p)
        x = tan(theta)
        y = - tan(phi) * sqrt(pow(tan(theta), 2) + 1)
        return np.array([x, y, 1])

    @staticmethod
    def from_relative_3d_to_2d(u, v, f, p, t, pos):
        """
        from 3d camera coordinate to image (The function of K*Q_tilt*Q_pan)
        :param u: camera parameter u
        :param v: camera parameter v
        :param f: camera parameter f
        :param p: pan
        :param t: tilt
        :param pos: 3-d point in camera coordinate: array [3]
        :return: 2d point tuple (x,y) in image
        """
        pan = radians(p)
        tilt = radians(t)

        k = np.array([[f, 0, u],
                      [0, f, v],
                      [0, 0, 1]])

        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),

                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))

        position = np.dot(k, np.dot(rotation, pos))

        return position[0] / position[2], position[1] / position[2]

    @staticmethod
    def from_3d_to_relative_3d(c, base_r, pos):
        """
        from 3d world coordinate to camera coordinate (The function of S[I|-C])
        :param c: projection center: array [3]
        :param base_r: base rotation matrix: array [3, 3]
        :param pos: 3-d point in world coordinate: array [3]
        :return: 3d point in camera coordinate: array [3]
        """
        position = np.dot(base_r, pos - c)
        return position / position[2]

    @staticmethod
    def get_observation_from_rays(pan, tilt, f, rays, u, v, height=0, width=0):
        """
        from a number of points to corresponding rays im image.
        :param pan: camera pan
        :param tilt: camera tilt
        :param f: camera f
        :param rays: [N, 2] array
        :param u: camera parameter
        :param v: camera parameter
        :param height: image height
        :param width: image width
        :return: 2-d points: [n, 2] array, indexes: [n] array (indexes of points in image)
        """
        points = np.ndarray([0, 2])
        index = np.ndarray([0])

        if height != 0 and width != 0:
            for j in range(len(rays)):
                tmp = TransFunction.from_pan_tilt_to_2d(u, v, f, pan, tilt, rays[j][0], rays[j][1])
                if 0 < tmp[0] < width and 0 < tmp[1] < height:
                    points = np.row_stack([points, np.asarray(tmp)])
                    index = np.concatenate([index, [j]], axis=0)
        else:
            for j in range(len(rays)):
                tmp = TransFunction.from_pan_tilt_to_2d(u, v, f, pan, tilt, rays[j][0], rays[j][1])
                points = np.row_stack([points, np.asarray(tmp)])

        return points, index

    @staticmethod
    def get_rays_from_observation(pan, tilt, f, points, u, v):
        """
        get a list of rays from 2d points and camera pose
        :param pan:
        :param tilt:
        :param f:
        :param points: [PointNumber, 2]
        :param u: camera parameter
        :param v: camera parameter
        :return: [RayNumber(=PointNumber), 2]
        """
        rays = np.ndarray([0, 2])
        for i in range(len(points)):
            angles = TransFunction.from_2d_to_pan_tilt(u, v, f, pan, tilt, points[i][0], points[i][1])
            rays = np.row_stack([rays, angles])
        return rays

    """below is function for general slam"""

    @staticmethod
    def get_observation_from_3ds(pan, tilt, f, rays, u, v, center, rotation, height=0, width=0):
        """
        from a number of points to corresponding rays im image.
        :param pan: camera pan
        :param tilt: camera tilt
        :param f: camera f
        :param rays: [N, 2] array
        :param u: camera parameter
        :param v: camera parameter
        :param height: image height
        :param width: image width
        :return: 2-d points: [n, 2] array, indexes: [n] array (indexes of points in image)
        """
        points = np.ndarray([0, 2])
        index = np.ndarray([0])

        if height != 0 and width != 0:
            for j in range(len(rays)):
                tmp = TransFunction.from_3d_to_2d(u, v, f, pan, tilt, center, rotation, rays[j])
                if 0 < tmp[0] < width and 0 < tmp[1] < height:
                    points = np.row_stack([points, np.asarray(tmp)])
                    index = np.concatenate([index, [j]], axis=0)
        else:
            for j in range(len(rays)):
                tmp = TransFunction.from_3d_to_2d(u, v, f, pan, tilt, center, rotation, rays[j])
                points = np.row_stack([points, np.asarray(tmp)])

        return points, index

    @staticmethod
    def get_3ds_from_observation(pan, tilt, f, points, u, v, center, rotation):
        """
        get a list of rays from 2d points and camera pose
        :param pan:
        :param tilt:
        :param f:
        :param points: [PointNumber, 2]
        :param u: camera parameter
        :param v: camera parameter
        :return: [RayNumber(=PointNumber), 2]
        """
        rays = np.ndarray([0, 3])
        for i in range(len(points)):
            position = TransFunction.from_2d_to_3d(u, v, f, pan, tilt, center, rotation, points[i])
            rays = np.row_stack([rays, position])

        return rays
