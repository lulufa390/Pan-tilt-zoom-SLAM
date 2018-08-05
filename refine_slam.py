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
        1. load the soccer field model
        2. load the sequence annotation
        3. load the synthesized data
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
        get the first camera pose 
        f, pan, tilt are 3 variables for camera pose
        u, v are the image center
        """

        """
        base_rotation: the base rotation matrix of PTZ camera         
        c: projection center of ptz camera
        """
        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = self.meta[0][0]["cc"][0]

        """
        1. initialize the camera pose array 
        2. initialize the covariance matrix array 
        3. speed model for PTZ camera
        """
        self.x = np.ndarray([self.annotation.size, 3])
        pan, tilt, f = self.annotation[0][0]['ptz'].squeeze()
        self.x[0] = [pan * math.pi / 180, tilt * math.pi / 180, f]

        self.p = np.ndarray([self.annotation.size, 3, 3])
        self.p[0] = np.diag([0.01, 0.01, 0.01])

        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

        self.q = 0.01 * np.ones([3, 3])
        self.r = 0.01 * np.ones([3, 3])

    @staticmethod
    def compute_jacobi(theta, phi, foc, ray):
        jacobi_h = np.ndarray([2, 3])
        jacobi_h[0][0] = -foc / math.pow(math.cos(ray[0] - theta), 2)
        jacobi_h[0][1] = 0
        jacobi_h[0][2] = math.tan(ray[0] - theta)
        jacobi_h[1][0] = 0
        jacobi_h[1][1] = foc / math.pow(math.cos(ray[1] - phi), 2)
        jacobi_h[1][2] = -math.tan(ray[1] - phi)
        return jacobi_h

    @staticmethod
    def compute_jacobi_approximate(theta, phi, foc, ray):
        jacobi_h = np.ndarray([2, 3])
        delta = 0.01
        jacobi_h[0][0] = (foc * math.tan(ray[0] - (theta + delta)) - foc * math.tan(ray[0] - (theta - delta))) / (
                2 * delta)
        jacobi_h[0][1] = 0
        jacobi_h[0][2] = ((foc + delta) * math.tan(ray[0] - theta) - (foc - delta) * math.tan(ray[0] - theta)) / (
                2 * delta)
        jacobi_h[1][0] = 0
        jacobi_h[1][1] = (-foc * math.tan(ray[1] - (phi + delta)) + foc * math.tan(ray[1] - (phi - delta))) / (
                2 * delta)
        jacobi_h[1][2] = ((foc + delta) * math.tan(ray[1] - phi) - (foc - delta) * math.tan(ray[1] - phi)) / (2 * delta)

        return jacobi_h

    def ekf(self, previous_x, previous_p, observe):
        predict_x = previous_x + [self.delta_pan, self.delta_tilt, self.delta_zoom]

        predict_p = previous_p + self.q

        hx = np.ndarray([len(self.rays), 2])

        for j in range(len(self.rays)):
            hx[j][0] = predict_x[2] * math.tan(self.rays[j][0] - predict_x[0]) + self.u
            hx[j][1] = - predict_x[2] * math.tan(self.rays[j][1] - predict_x[1]) + self.v
            # cv.circle(self.img, (int(hx[j]  [0]), int(hx[j][1])), color=(255, 0, 0), radius=8, thickness=2)

        y = []
        jacobi = []
        cnt = 0
        for j in range(len(self.rays)):
            if 0 < observe[j][0] < 1280 and 0 < observe[j][1] < 720 and 0 < hx[j][0] < 1280 and 0 < hx[j][1] < 720:
                y.append(observe[j][0] - hx[j][0])
                y.append(observe[j][1] - hx[j][1])
                jacobi.append(PtzSlam.compute_jacobi(predict_x[0], predict_x[1], predict_x[2], self.rays[j]))
                cnt += 1

        y = np.array(y)
        jacobi = np.array(jacobi)
        jacobi = jacobi.reshape((-1, 3))



        s = np.dot(np.dot(jacobi, predict_p), jacobi.T) + 0.01 * np.ones([2 * cnt, 2 * cnt])

        k = np.dot(np.dot(predict_p, jacobi.T), np.linalg.inv(s))

        updated_x = np.transpose(predict_x + np.dot(k, y))
        updated_p = np.dot((np.eye(3) - np.dot(k, jacobi)), predict_p)
        return [updated_x, updated_p]




    def get_observation(self, index):
        f = self.annotation[0][index]['camera'][0][2]
        pan, tilt, _ = self.annotation[0][index]['ptz'].squeeze() * math.pi / 180

        features = np.ndarray([len(self.rays), 2])

        for j in range(len(self.rays)):
            pos = np.array(self.pts[j])
            features[j] = synthesize.from_3d_to_2d(self.u, self.v, f, pan, tilt, self.c, self.base_rotation, pos)
            cv.circle(self.img, (int(features[j][0]), int(features[j][1])), color=(0, 0, 0), radius=8, thickness=2)

        return features

    def main_algorithm(self):
        for i in range(1, self.annotation.size):

            self.img.fill(255)

            features = self.get_observation(i)

            """
            yk = zk - h(xk|k-1)       
            """

            self.x[i], self.p[i] = self.ekf(self.x[i-1], self.p[i-1], features)

            self.delta_pan, self.delta_tilt, self.delta_zoom = self.x[i] - self.x[i - 1]

            cv.imshow("synthesized image", self.img)

            cv.waitKey(0)

        print(self.x)

    # for i in range(1, 15):
    #     for j in range(1, 15):
    #         print("theta:", i * 0.1, "phi", j * 0.1)
    #         print(compute_jacobi(0.9, 0.1, 3500, [i * 0.1, j * 0.1]))
    #
    #         print(compute_jacobi_approximate(0.9, 0.1, 3500, [i * 0.1, j * 0.1]))


slam = PtzSlam("./two_point_calib_dataset/util/highlights_soccer_model.mat",
               "./two_point_calib_dataset/highlights/seq3_anno.mat",
               "./synthesize_data.mat")
slam.main_algorithm()
