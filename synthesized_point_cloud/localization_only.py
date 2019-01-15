"""
PTZ camera SLAM tested on synthesized data
2018.8
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
from sklearn.preprocessing import normalize
from math import *
from transformation import TransFunction


class PtzSlam:
    def __init__(self, model_path, annotation_path, data_path):
        self.img = np.zeros((720, 1280, 3), np.uint8)

        """
        load the data of soccer field
        load annotation (ground truth)
        load synthesized features
        """
        soccer_model = sio.loadmat(model_path)
        self.line_index = soccer_model['line_segment_index']
        self.points = soccer_model['points']

        seq = sio.loadmat(annotation_path)
        self.annotation = seq["annotation"]
        self.meta = seq['meta']

        data = sio.loadmat(data_path)
        self.pts = data["pts"]
        self.features = data["features"]
        self.rays = data["rays"]

        """
        initialize the fixed parameters of our algorithm
        u, v, base_rotation and c
        """
        self.u, self.v, self.base_rotation, self.c = self.get_u_v_rotation_center()

        """
        parameters to be updated
        """
        self.x = np.ndarray([self.annotation.size, 3])
        self.x[0] = self.get_ptz(0)

        self.p = np.ndarray([self.annotation.size, 3, 3])
        self.p[0] = np.diag([0.01, 0.01, 0.01])

        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

    def get_ptz(self, index):
        return self.annotation[0][index]['ptz'].squeeze()

    def get_u_v_rotation_center(self):
        u, v = self.annotation[0][0]['camera'][0][0:2]
        base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], base_rotation)
        camera_center = self.meta[0][0]["cc"][0]
        return u, v, base_rotation, camera_center

    def compute_new_jacobi(self, camera_pan, camera_tilt, foc, ray):
        jacobi_h = np.ndarray([2, 3])
        delta_angle = 0.001
        delta_f = 1.0

        # c_pan = radians(camera_pan)
        # c_tilt = radians(camera_tilt)
        # theta = radians(ray[0])
        # phi = radians(ray[1])

        x_delta_pan1, y_delta_pan1 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan - delta_angle,
                                                                       camera_tilt, ray[0], ray[1])
        x_delta_pan2, y_delta_pan2 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan + delta_angle, camera_tilt,
                                                                       ray[0], ray[1])

        x_delta_tilt1, y_delta_tilt1 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan, camera_tilt - delta_angle,
                                                                         ray[0], ray[1])
        x_delta_tilt2, y_delta_tilt2 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan, camera_tilt + delta_angle,
                                                                         ray[0], ray[1])

        x_delta_f1, y_delta_f1 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc - delta_f, camera_pan,
                                                                   camera_tilt, ray[0], ray[1])
        x_delta_f2, y_delta_f2 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc + delta_f, camera_pan
                                                                   , camera_tilt, ray[0], ray[1])

        jacobi_h[0][0] = (x_delta_pan2 - x_delta_pan1) / (2 * delta_angle)
        jacobi_h[0][1] = (x_delta_tilt2 - x_delta_tilt1) / (2 * delta_angle)
        jacobi_h[0][2] = (x_delta_f2 - x_delta_f1) / (2 * delta_f)

        jacobi_h[1][0] = (y_delta_pan2 - y_delta_pan1) / (2 * delta_angle)
        jacobi_h[1][1] = (y_delta_tilt2 - y_delta_tilt1) / (2 * delta_angle)
        jacobi_h[1][2] = (y_delta_f2 - y_delta_f1) / (2 * delta_f)

        return jacobi_h

    def get_observation_from_ptz(self, pan, tilt, f):
        points = np.ndarray([len(self.rays), 2])
        for j in range(len(self.rays)):
            pos = np.array(self.pts[j])
            points[j] = TransFunction.from_3d_to_2d(self.u, self.v, f, pan, tilt, self.c,
                                      self.base_rotation, pos)
        return points

    def visualize_points(self, points, pt_color):
        for j in range(len(points)):
            cv.circle(self.img, (int(points[j][0]), int(points[j][1])), color=pt_color, radius=8, thickness=2)

    def get_points_in_image(self, observe, predict, width, height):
        index_list = []
        for i in range(len(self.rays)):
            if 0 < observe[i][0] < width and 0 < observe[i][1] < height and 0 < predict[i][0] < width and 0 < \
                    predict[i][1] < height:
                index_list.append(i)
        return index_list

    def extended_kalman_filter(self, previous_x, previous_p, z_k):

        # 1. predict step
        predict_x = previous_x + [self.delta_pan, self.delta_tilt, self.delta_zoom]

        print("\n-----predict_x-----\n", predict_x)

        q_k = 1.0 * np.diag([0.01, 0.01, 1])
        predict_p = previous_p + q_k

        # 2. update step
        hx = np.ndarray([len(self.rays), 2])
        for j in range(len(self.rays)):
            hx[j] = TransFunction.from_pan_tilt_to_2d(self.u, self.v, predict_x[2], predict_x[0], predict_x[1],
                                        self.rays[j][0], self.rays[j][1])

        print("\n-----hx-----\n", hx)

        inner_index = self.get_points_in_image(z_k, hx, 1280, 720)

        y = np.ndarray([2 * len(inner_index), 1])
        jacobi = np.ndarray([2 * len(inner_index), 3])
        for i in range(len(inner_index)):
            pts_index = inner_index[i]
            y[2 * i] = z_k[pts_index][0] - hx[pts_index][0]
            y[2 * i + 1] = z_k[pts_index][1] - hx[pts_index][1]
            jacobi[2 * i:2 * i + 2, 0:3] = self.compute_new_jacobi(predict_x[0], predict_x[1], predict_x[2],
                                                                   self.rays[pts_index])

        print("\n-----jacobi-----\n", jacobi)

        print("\n-----H*P*H^t-----\n", np.dot(np.dot(jacobi, predict_p), jacobi.T))

        s = np.dot(np.dot(jacobi, predict_p), jacobi.T) + 0.1 * np.eye(2 * len(inner_index))
        k = np.dot(np.dot(predict_p, jacobi.T), np.linalg.inv(s))

        print("\n-----y-----\n", y.shape)
        print("\n-----k-----\n", k.shape)
        print("\n-----k*y-----\n", np.dot(k, y))
        updated_x = predict_x + np.dot(k, y).squeeze()
        updated_p = np.dot((np.eye(3) - np.dot(k, jacobi)), predict_p)

        return updated_x, updated_p

    def main_algorithm(self):

        self.img.fill(255)
        # for i in range(1, 2):
        for i in range(1, self.annotation.size):
            self.img.fill(255)

            next_pan, next_tilt, next_f = self.get_ptz(i)
            observe = self.get_observation_from_ptz(next_pan, next_tilt, next_f)

            self.x[i], self.p[i] = self.extended_kalman_filter(self.x[i - 1], self.p[i - 1], observe)
            self.delta_pan, self.delta_tilt, self.delta_zoom = self.x[i] - self.x[i - 1]

            estimate_features = self.get_observation_from_ptz(self.x[i][0], self.x[i][1], self.x[i][2])
            predict_features = self.get_observation_from_ptz(self.x[i - 1][0], self.x[i - 1][1], self.x[i - 1][2])
            self.visualize_points(observe, (0, 0, 0))
            self.visualize_points(estimate_features, (0, 0, 255))
            self.visualize_points(predict_features, (255, 0, 0))

            cv.imshow("synthesized image", self.img)
            cv.waitKey(0)

            # self.x[i] = self.get_ptz(i)

            # self.delta_pan, self.delta_tilt, self.delta_zoom = [0,0,0]
            #
            # index = self.get_points_in_image(estimate_features, observe, 1280, 720)
            # norm = 0
            # for j in range(len(index)):
            #     norm += np.linalg.norm(estimate_features[index[j]] - observe[index[j]])
            # return np.linalg.norm(estimate_features - observe)


slam = PtzSlam("./two_point_calib_dataset/util/highlights_soccer_model.mat",
               "./two_point_calib_dataset/highlights/seq3_anno.mat",
               "./synthesize_data.mat")

slam.main_algorithm()
