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
import math
import synthesize


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
        self.p[0] = np.diag([100, 100, 1])

        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

        self.q = 0.01* np.diag([1000, 1000, 1])
        # self.r = 0.01 * np.ones([3, 3])

    @staticmethod
    def compute_jacobi(theta, phi, foc, ray):
        jacobi_h = np.ndarray([2, 3])
        jacobi_h[0][0] = -foc / math.pow(math.cos( math.radians(ray[0] - theta) ), 2)
        jacobi_h[0][1] = 0
        jacobi_h[0][2] = math.tan(math.radians(ray[0] - theta))
        jacobi_h[1][0] = 0
        jacobi_h[1][1] = foc / math.pow(math.cos(math.radians(ray[1] - phi)), 2)
        jacobi_h[1][2] = -math.tan(math.radians(ray[1] - phi))
        return jacobi_h

    @staticmethod
    def compute_jacobi_approximate(theta, phi, foc, ray):
        jacobi_h = np.ndarray([2, 3])
        delta = 0.0001
        jacobi_h[0][0] = (foc * math.tan(ray[0] - (theta + delta)) - foc * math.tan(ray[0] - (theta - delta))) / (
                2 * delta)
        jacobi_h[0][1] = 0
        jacobi_h[0][2] = ((foc + delta) * math.tan(ray[0] - theta) - (foc - delta) * math.tan(ray[0] - theta)) / (
                2 * delta)
        jacobi_h[1][0] = 0
        jacobi_h[1][1] = (-foc * math.tan(ray[1] - (phi + delta)) + foc * math.tan(ray[1] - (phi - delta))) / (
                2 * delta)
        jacobi_h[1][2] = (-(foc + delta) * math.tan(ray[1] - phi) + (foc - delta) * math.tan(ray[1] - phi)) / (2 * delta)
        return jacobi_h

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
        pan, tilt, _ = self.annotation[0][index]['ptz'].squeeze() * math.pi / 180
        features = np.ndarray([len(self.rays), 2])

        for j in range(len(self.rays)):
            pos = np.array(self.pts[j])
            features[j] = synthesize.from_3d_to_2d(self.u, self.v, f, pan, tilt, self.c, self.base_rotation, pos)

        return features

    def get_observation_from_ptz(self, pan, tilt, f):
        features = np.ndarray([len(self.rays), 2])
        for j in range(len(self.rays)):
            pos = np.array(self.pts[j])
            features[j] = synthesize.from_3d_to_2d(self.u, self.v, f, math.radians(pan), math.radians(tilt), self.c, self.base_rotation, pos)

        return features

    def visualize_features(self, features, pt_color):
        for j in range(len(features)):
            cv.circle(self.img, (int(features[j][0]), int(features[j][1])), color=pt_color, radius=8, thickness=2)

    def extended_kalman_filter(self, previous_x, previous_p, observe):
        predict_x = previous_x + [self.delta_pan, self.delta_tilt, self.delta_zoom]

        predict_p = previous_p + self.q

        hx = np.ndarray([len(self.rays), 2])

        for j in range(len(self.rays)):
            hx[j][0] = predict_x[2] * math.tan( math.radians(self.rays[j][0] - predict_x[0]) ) + self.u
            hx[j][1] = - predict_x[2] * math.tan( math.radians(self.rays[j][1] - predict_x[1]) ) + self.v

        y = []
        jacobi = []
        jacobi_a = []

        for j in range(len(self.rays)):
            if 0 < observe[j][0] < 1280 and 0 < observe[j][1] < 720 and 0 < hx[j][0] < 1280 and 0 < hx[j][1] < 720:
                y.append(observe[j][0] - hx[j][0])
                y.append(observe[j][1] - hx[j][1])
                jacobi.append(PtzSlam.compute_jacobi(predict_x[0], predict_x[1], predict_x[2], self.rays[j]))
                jacobi_a.append(PtzSlam.compute_jacobi_approximate(predict_x[0], predict_x[1], predict_x[2], self.rays[j]))

        y = np.array(y)
        jacobi = np.array(jacobi)
        jacobi = jacobi.reshape((-1, 3))

        jacobi_a = np.array(jacobi_a)
        jacobi_a = jacobi_a.reshape((-1, 3))

        # print(jacobi - jacobi_a)

        print(jacobi)
        # print(predict_p)

        # print(y)
        # print(jacobi)

        # test = np.array([[1,1,1], [1,2,3], [1,5,1]])
        # print(np.linalg.inv(test))


        # print(np.dot(np.dot(jacobi, predict_p), jacobi.T), "\n\n\n")
        s = np.dot(np.dot(jacobi, predict_p), jacobi.T) - 1 * np.ones([len(y), len(y)])

        # print(s)
        # print("inv s", np.linalg.inv(s))
        # print(y)



        k = np.dot(np.dot(predict_p, jacobi.T), np.linalg.inv(s))
        print(k)
        # print(k.shape)
        # print(y.shape)

        # print(np.dot(k,y) * 180 / math.pi)
        #
        # fuck = np.dot(k, y)
        # print(fuck.shape)
        # print(predict_x.shape)

        # print(predict_x + np.dot(k, y) )
        # updated_x = np.transpose(predict_x + np.dot(k, y))
        updated_x = predict_x + np.dot(k, y)
        updated_p = np.dot((np.eye(3) - np.dot(k, jacobi)), predict_p)
        return updated_x, updated_p

    def main_algorithm(self):
        self.img.fill(255)
        for i in range(1, 2):
        # for i in range(1, self.annotation.size):

            self.img.fill(255)

            observe = self.get_observation_from_index(i)
            self.x[i], self.p[i] = self.extended_kalman_filter(self.x[i - 1], self.p[i - 1], observe)
            self.delta_pan, self.delta_tilt, self.delta_zoom = self.x[i] - self.x[i - 1]

            # estimate_features = self.get_observation_from_ptz(self.x[i][0], self.x[i][1], self.x[i][2])
            # self.visualize_features(observe, (0, 0, 0))
            # self.visualize_features(estimate_features, (0, 0, 255))
            # cv.imshow("synthesized image", self.img)
            # cv.waitKey(0)

        features = self.get_observation_from_index(1)
        predict_features = self.get_observation_from_index(0)
        estimate_features = self.get_observation_from_ptz(self.x[1][0], self.x[1][1], self.x[1][2])


        self.visualize_features(predict_features, (255, 0, 0))
        self.visualize_features(features, (0, 0, 0))
        self.visualize_features(estimate_features, (0, 0, 255))

        print(self.x[1])
        print(self.annotation[0][1]['ptz'].squeeze())
        cv.imshow("synthesized image", self.img)
        cv.waitKey(0)


# PtzSlam.test_jacobi(0.9, 0.1, 3500, 0.7, 0.5)
PtzSlam.test_jacobi(0.9, 0.1, 3500, 1.5, 0.8)
PtzSlam.test_jacobi(0.9, 0.1, 3500, -0.6, 0.5)

slam = PtzSlam("./two_point_calib_dataset/util/highlights_soccer_model.mat",
               "./two_point_calib_dataset/highlights/seq3_anno.mat",
               "./synthesize_data.mat")
slam.main_algorithm()

