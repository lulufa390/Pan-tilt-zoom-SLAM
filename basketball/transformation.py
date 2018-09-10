"""
class 'TransFuntion' provides functions for 2d image projection and ray computing
All the input and output of functions in class 'TransFunction' is in degree!!

"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
from sklearn.preprocessing import normalize
from math import *


class TransFunction:
    @staticmethod
    def from_3d_to_2d(u, v, f, p, t, c, base_r, pos):
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
    def from_pan_tilt_to_2d(u, v, f, c_p, c_t, p, t):
        pan = radians(p)
        tilt = radians(t)
        camera_pan = radians(c_p)
        camera_tilt = radians(c_t)

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
        pan = radians(c_p)
        tilt = radians(c_t)

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
        relative = np.dot(base_r, np.transpose(pos - proj_center))
        x, y, z = relative
        theta = atan(x / z)
        phi = atan(-y / sqrt(x * x + z * z))

        return degrees(theta), degrees(phi)

    @staticmethod
    def from_ray_to_relative_3d(t, p):

        theta = radians(t)
        phi = radians(p)
        x = tan(theta)
        y = - tan(phi) * sqrt(pow(tan(theta), 2) + 1)
        return np.array([x,y,1])

    @staticmethod
    def from_relative_3d_to_2d(u, v, f, p, t, pos):
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
        position = np.dot(base_r, pos - c)
        return position / position[2]

