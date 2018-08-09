"""
PTZ camera SLAM tested on synthesized data
2018.8
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
from sklearn.preprocessing import normalize
from math import *
from synthesize import *


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
        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = self.meta[0][0]["cc"][0]

        """
        parameters to be updated
        """
        self.x = np.ndarray([self.annotation.size, 3])
        pan, tilt, f = self.annotation[0][0]['ptz'].squeeze()
        self.x[0] = [pan, tilt, f]

        self.p = np.ndarray([self.annotation.size, 3, 3])
        # self.p[0] = np.diag([1000, 1000, 1])
        self.p[0] = np.diag([0.01, 0.01, 0.01])

        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

        # self.q = 0.01 * np.diag([1000, 1000, 1])
        # self.r = 0.01 * np.ones([3, 3])

    @staticmethod
    def compute_jacobi(theta, phi, foc, ray):
        jacobi_h = np.ndarray([2, 3])
        jacobi_h[0][0] = -foc / pow(cos(radians(ray[0] - theta)), 2)
        jacobi_h[0][1] = 0
        jacobi_h[0][2] = tan(radians(ray[0] - theta))
        jacobi_h[1][0] = 0
        jacobi_h[1][1] = foc / pow(cos(radians(ray[1] - phi)), 2)
        jacobi_h[1][2] = -tan(radians(ray[1] - phi))
        return jacobi_h

    @staticmethod
    def compute_jacobi_approximate(theta, phi, foc, ray):
        jacobi_h = np.ndarray([2, 3])
        delta = 0.0001
        jacobi_h[0][0] = (foc * tan(ray[0] - (theta + delta)) - foc * tan(ray[0] - (theta - delta))) / (
                2 * delta)
        jacobi_h[0][1] = 0
        jacobi_h[0][2] = ((foc + delta) * tan(ray[0] - theta) - (foc - delta) * tan(ray[0] - theta)) / (
                2 * delta)
        jacobi_h[1][0] = 0
        jacobi_h[1][1] = (-foc * tan(ray[1] - (phi + delta)) + foc * tan(ray[1] - (phi - delta))) / (
                2 * delta)
        jacobi_h[1][2] = (-(foc + delta) * tan(ray[1] - phi) + (foc - delta) * tan(ray[1] - phi)) / (
                2 * delta)
        return jacobi_h


    @staticmethod
    def compute_new_jacobi(camera_pan, camera_tilt, foc, ray):
        jacobi_h = np.ndarray([2, 3])
        delta = 0.0001

        theta, phi = compute_pose_relative_angles(ray[0], ray[1], camera_pan, camera_tilt)

        theta_delta_pan1, phi_delta_pan1 = compute_pose_relative_angles(ray[0], ray[1], camera_pan - delta, camera_tilt)
        theta_delta_pan2, phi_delta_pan2 = compute_pose_relative_angles(ray[0], ray[1], camera_pan + delta, camera_tilt)

        theta_delta_tilt1, phi_delta_tilt1 = compute_pose_relative_angles(ray[0], ray[1], camera_pan,
                                                                          camera_tilt - delta)
        theta_delta_tilt2, phi_delta_tilt2 = compute_pose_relative_angles(ray[0], ray[1], camera_pan,
                                                                          camera_tilt + delta)

        jacobi_h[0][0] = (foc * tan(theta_delta_pan2) - foc * tan(theta_delta_pan1)) / (2 * delta)
        jacobi_h[0][1] = (foc * tan(theta_delta_tilt2) - foc * tan(theta_delta_tilt1)) / (2 * delta)
        jacobi_h[0][2] = ((foc + delta) * tan(theta) - (foc - delta) * tan(theta)) / (2 * delta)

        jacobi_h[1][0] = ((-foc * sqrt(1 + pow(tan(theta_delta_pan2), 2) ) * tan(phi_delta_pan2)) - (
                    -foc * sqrt(1 + pow(tan(theta_delta_pan1), 2) )* tan(phi_delta_pan1))) / (2 * delta)

        jacobi_h[1][1] = ((-foc * sqrt(1 + pow(tan(theta_delta_tilt2), 2) ) * tan(phi_delta_tilt2)) - (
                -foc * sqrt(1 + pow(tan(theta_delta_tilt1), 2) )* tan(phi_delta_tilt1))) / (2 * delta)
        jacobi_h[1][2] = ((-(foc+delta) * sqrt(1 + pow(tan(theta), 2) )* tan(phi)) - (
                -(foc-delta) * sqrt(1 + pow(tan(theta), 2) )* tan(phi))) / (2 * delta)

        return jacobi_h

        # phi_delta_pan1 = compute_pose_relative_angles(ray[0], ray[1], camera_pan - delta, camera_tilt)

        # pan1 = compute_pose_relative_angles(ray[0], ray[1], theta - delta, phi)
        # pan2 = compute_pose_relative_angles(ray[0], ray[1], theta + delta, phi)
        # jacobi_h[0][0] = ( foc * tan(pan2) - foc * tan(pan1) ) / (2*delta)
        #
        # pan1 = compute_pose_relative_angles(ray[0], ray[1], theta, phi - delta)
        # pan2 = compute_pose_relative_angles(ray[0], ray[1], theta, phi + delta)
        # jacobi_h[0][1] = (foc * tan(pan2) - foc * tan(pan1)) / (2 * delta)


    @staticmethod
    def test_jacobi(theta, phi, f, theta_i, phi_i):
        print("----------------------------------------")
        print("theta = %f, phi = %f, f = %f, theta_i = %f, phi_i = %f" % (theta, phi, f, theta_i, phi_i), "\n")
        print("result of derivative:")
        print(PtzSlam.compute_jacobi(theta, phi, f, [theta_i, phi_i]), "\n")
        print("result of approximate:")
        print(PtzSlam.compute_jacobi_approximate(theta, phi, f, [theta_i, phi_i]))
        print("----------------------------------------\n")

    def get_observation_from_index(self, index):
        f = self.annotation[0][index]['camera'][0][2]
        pan, tilt, _ = self.annotation[0][index]['ptz'].squeeze() * pi / 180
        features = np.ndarray([len(self.rays), 2])

        for j in range(len(self.rays)):
            pos = np.array(self.pts[j])
            features[j] = from_3d_to_2d(self.u, self.v, f, pan, tilt, self.c, self.base_rotation, pos)

        return features

    def get_observation_from_ptz(self, pan, tilt, f):
        features = np.ndarray([len(self.rays), 2])
        for j in range(len(self.rays)):
            pos = np.array(self.pts[j])
            features[j] = from_3d_to_2d(self.u, self.v, f, radians(pan), radians(tilt), self.c,
                                                   self.base_rotation, pos)

        return features

    def visualize_features(self, features, pt_color):
        for j in range(len(features)):
            cv.circle(self.img, (int(features[j][0]), int(features[j][1])), color=pt_color, radius=8, thickness=2)

    def get_points_in_image(self, observe, predict, width, height):
        index_list = []
        for i in range(len(self.rays)):
            if 0 < observe[i][0] < width and 0 < observe[i][1] < height and 0 < predict[i][0] < width and 0 < \
                    predict[i][1] < height:
                index_list.append(i)
        return index_list



    def extended_kalman_filter(self, previous_x, previous_p, observe):
        predict_x = previous_x + [self.delta_pan, self.delta_tilt, self.delta_zoom]

        predict_p = previous_p + self.q * np.diag([self.scale, self.scale, 1])

        hx = np.ndarray([len(self.rays), 2])

        for j in range(len(self.rays)):
            hx[j][0] = predict_x[2] * tan(radians(self.rays[j][0] - predict_x[0])) + self.u
            hx[j][1] = - predict_x[2] * tan(radians(self.rays[j][1] - predict_x[1])) + self.v

        index = self.get_points_in_image(observe, hx, 1280, 720)
        y = np.ndarray([2 * len(index), 1])
        jacobi = np.ndarray([2 * len(index), 3])
        for i in range(len(index)):
            pts_index = index[i]
            y[2 * i] = observe[pts_index][0] - hx[pts_index][0]
            y[2 * i + 1] = observe[pts_index][1] - hx[pts_index][1]

            jacobi[2 * i:2 * i + 2, 0:3] = PtzSlam.compute_new_jacobi(predict_x[0], predict_x[1], predict_x[2],
                                                                  self.rays[pts_index])

        print(jacobi)

        # s = np.dot(np.dot(jacobi, predict_p), jacobi.T) - self.r * np.ones([2 * len(index), 2 * len(index)])
        s = np.dot(np.dot(jacobi, predict_p), jacobi.T) - self.r * np.eye(2 * len(index))
        k = np.dot(np.dot(predict_p, jacobi.T), np.linalg.inv(s))


        # print("y", y, "\n\n")
        updated_x = predict_x + np.dot(k, y).squeeze()
        updated_p = np.dot((np.eye(3) - np.dot(k, jacobi)), predict_p)

        return updated_x, updated_p

    def main_algorithm(self, q, r , scale):
        self.q = q
        self.r = r
        self.scale = scale

        self.img.fill(255)
        for i in range(1, 2):
        # for i in range(1, self.annotation.size):

            self.img.fill(255)

            observe = self.get_observation_from_index(i)

            # print("predict before: ", self.x[i - 1])

            self.x[i], self.p[i] = self.extended_kalman_filter(self.x[i - 1], self.p[i - 1], observe)
            self.delta_pan, self.delta_tilt, self.delta_zoom = self.x[i] - self.x[i - 1]

            # print("predict: ", self.x[i-1])


            # print("update", self.x[i])

            # print("ground truth", np.transpose(self.annotation[0][i]['ptz'] ), "\n\n")

            estimate_features = self.get_observation_from_ptz(self.x[i][0], self.x[i][1], self.x[i][2])
            predict_features = self.get_observation_from_ptz(self.x[i-1][0], self.x[i-1][1], self.x[i-1][2])
            self.visualize_features(observe, (0, 0, 0))
            self.visualize_features(estimate_features, (0, 0, 255))
            self.visualize_features(predict_features, (255, 0, 0))

            cv.imshow("synthesized image", self.img)
            cv.waitKey(0)

            # self.delta_pan, self.delta_tilt, self.delta_zoom = [0,0,0]
            #
            # index = self.get_points_in_image(estimate_features, observe, 1280, 720)
            # norm = 0
            # for j in range(len(index)):
            #     norm += np.linalg.norm(estimate_features[index[j]] - observe[index[j]])
            # return np.linalg.norm(estimate_features - observe)



        # features = self.get_observation_from_index(1)
        # predict_features = self.get_observation_from_index(0)
        # estimate_features = self.get_observation_from_ptz(self.x[1][0], self.x[1][1], self.x[1][2])

        # self.visualize_features(predict_features, (255, 0, 0))
        # self.visualize_features(features, (0, 0, 0))
        # self.visualize_features(estimate_features, (0, 0, 255))

        # print(self.x[1])
        # print(self.annotation[0][1]['ptz'].squeeze())
        # cv.imshow("synthesized image", self.img)
        # cv.waitKey(0)




# PtzSlam.test_jacobi(0.9, 0.1, 3500, 0.7, 0.5)
# PtzSlam.test_jacobi(0.9, 0.1, 3500, 1.5, 0.8)
PtzSlam.test_jacobi(0.9, 0.1, 3500, -0.6, 0.5)

slam = PtzSlam("./two_point_calib_dataset/util/highlights_soccer_model.mat",
               "./two_point_calib_dataset/highlights/seq3_anno.mat",
               "./synthesize_data.mat")

# min_norm = math.inf
# min_r = -1
# min_q = -1
#
# for r in np.arange(-1, 1 , 0.1):
#     for q in np.arange(-1, 1, 0.1):
#         # for s in np.arange(0, 1, 0.01):
#         tmp_norm = slam.main_algorithm(q, r,100)
#
#         if tmp_norm < min_norm:
#             min_norm, min_q, min_r = tmp_norm, q, r
#             print(min_norm)
#
#     # print(r)
#
# print("min", min_q, min_r)

slam.main_algorithm(-0.7, -0.2, 100)


