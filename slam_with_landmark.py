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
        self.all_rays = np.column_stack((data["rays"], data["features"]))


        """
        initialize the fixed parameters of our algorithm
        u, v, base_rotation and c
        """
        self.u, self.v, self.base_rotation, self.c = self.get_u_v_rotation_center()

        """
        parameters to be updated
        """
        self.camera_pose = np.ndarray([3])
        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

        self.ray_global = np.ndarray([0, self.all_rays.shape[1]])
        self.p_global = np.zeros([3, 3])


    def get_ptz(self, index):
        return self.annotation[0][index]['ptz'].squeeze()

    def get_u_v_rotation_center(self):
        u, v = self.annotation[0][0]['camera'][0][0:2]
        base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], base_rotation)
        camera_center = self.meta[0][0]["cc"][0]
        return u, v, base_rotation, camera_center

    def compute_new_jacobi(self, camera_pan, camera_tilt, foc, rays):
        ray_num = len(rays)

        delta_angle = 0.001
        delta_f = 1.0

        jacobi_h = np.ndarray([2 * ray_num, 3 + 2 * ray_num])

        for i in range(ray_num):
            x_delta_pan1, y_delta_pan1 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan - delta_angle,
                                                                              camera_tilt, rays[i][0], rays[i][1])
            x_delta_pan2, y_delta_pan2 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan + delta_angle, camera_tilt,
                                                                           rays[i][0], rays[i][1])

            x_delta_tilt1, y_delta_tilt1 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan, camera_tilt - delta_angle,
                                                                             rays[i][0], rays[i][1])
            x_delta_tilt2, y_delta_tilt2 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan, camera_tilt + delta_angle,
                                                                             rays[i][0], rays[i][1])

            x_delta_f1, y_delta_f1 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc - delta_f, camera_pan,
                                                                       camera_tilt, rays[i][0], rays[i][1])
            x_delta_f2, y_delta_f2 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc + delta_f, camera_pan
                                                                       , camera_tilt, rays[i][0], rays[i][1])

            x_delta_theta1, y_delta_theta1 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan,
                                                                       camera_tilt, rays[i][0] - delta_angle, rays[i][1])
            x_delta_theta2, y_delta_theta2 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan
                                                                       , camera_tilt, rays[i][0] + delta_angle, rays[i][1])

            x_delta_phi1, y_delta_phi1 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan,
                                                                       camera_tilt, rays[i][0], rays[i][1] - delta_angle)
            x_delta_phi2, y_delta_phi2 = TransFunction.from_pan_tilt_to_2d(self.u, self.v, foc, camera_pan
                                                                       , camera_tilt, rays[i][0] , rays[i][1] + delta_angle)

            jacobi_h[2*i][0] = (x_delta_pan2 - x_delta_pan1) / (2 * delta_angle)
            jacobi_h[2*i][1] = (x_delta_tilt2 - x_delta_tilt1) / (2 * delta_angle)
            jacobi_h[2*i][2] = (x_delta_f2 - x_delta_f1) / (2 * delta_f)

            jacobi_h[2*i+1][0] = (y_delta_pan2 - y_delta_pan1) / (2 * delta_angle)
            jacobi_h[2*i+1][1] = (y_delta_tilt2 - y_delta_tilt1) / (2 * delta_angle)
            jacobi_h[2*i+1][2] = (y_delta_f2 - y_delta_f1) / (2 * delta_f)

            for j in range(ray_num):
                if j == i:
                    jacobi_h[2 * i][3 + 2*j] = (x_delta_theta2 - x_delta_theta1) / (2 * delta_angle)
                    jacobi_h[2 * i][3 + 2*j +1] = (x_delta_phi2 - x_delta_phi1) / (2 * delta_angle)

                    jacobi_h[2 * i+1][3 + 2 * j] = (y_delta_theta2 - y_delta_theta1) / (2 * delta_angle)
                    jacobi_h[2 * i+1][3 + 2 * j + 1] = (y_delta_phi2 - y_delta_phi1) / (2 * delta_angle)
                else:
                    jacobi_h[2 * i][3 + 2 * j] = jacobi_h[2 * i][3 + 2 * j + 1] = \
                    jacobi_h[2 * i + 1][3 + 2 * j] = jacobi_h[2 * i + 1][3 + 2 * j + 1] = 0

        return jacobi_h

    def get_observation_from_ptz(self, pan, tilt, f, rays, width, height):
        points = np.ndarray([0, 2])
        inner_rays = np.ndarray([0, rays.shape[1]])
        index = np.ndarray([0])

        for j in range(len(rays)):
            tmp = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, rays[j][0], rays[j][1])
            if 0 < tmp[0] < width and 0 < tmp[1] < height:
                inner_rays = np.row_stack((inner_rays, rays[j]))
                points = np.row_stack((points, tmp))
                index = np.concatenate((index, [j]), axis=0)
        return points, inner_rays, index


    def visualize_points(self, points, pt_color):
        for j in range(len(points)):
            cv.circle(self.img, (int(points[j][0]), int(points[j][1])), color=pt_color, radius=8, thickness=2)


    def main_algorithm(self):

        self.img.fill(255)

        self.camera_pose = self.get_ptz(0)
        points, in_rays, index = self.get_observation_from_ptz(self.camera_pose[0], self.camera_pose[1],
                                                   self.camera_pose[2],self.all_rays, 1280, 720)

        # add rays in frame 1 to global rays
        self.ray_global = np.concatenate([self.ray_global, in_rays], axis=0)

        # initialize global p using global rays
        self.p_global = 0.001 * np.eye(3+2*len(self.ray_global))
        self.p_global[2][2] = 0.1
        # print(self.p_global)


        for i in range(1, 2):
            # ground truth for next frame. In real data do not need to compute
            next_pan, next_tilt, next_f = self.get_ptz(i)
            points_next, in_rays_next, _ = self.get_observation_from_ptz(next_pan, next_tilt, next_f, self.all_rays, 1280, 720)


            self.camera_pose += [self.delta_pan, self.delta_tilt, self.delta_zoom]
            points_before, in_rays_before, index_before = self.get_observation_from_ptz(self.camera_pose[0], self.camera_pose[1],
                                                                          self.camera_pose[2], self.ray_global, 1280, 720)

            y_k = np.ndarray([0])
            for j in range(len(in_rays_before)):
                flag = False
                for k in range(len(in_rays_next)):
                    if np.linalg.norm(in_rays_next[k][2:18] - in_rays_before[j][2:18]) < 0.01:

                        y_k = np.concatenate((y_k, points_next[k] - points_before[j]), axis=0)
                        flag = True
                if not flag:
                    index_before = np.delete(index_before, j, axis=0)


            predict_x = self.camera_pose
            for j in range(len(index_before)):
                predict_x = np.concatenate([predict_x, self.ray_global[int(index_before[j])][0:2]], axis=0)

            p_index = (np.concatenate([[0,1,2], index_before+3, index_before+len(index_before) + 3])).astype(int)
            # print("fuck", index_before)
            p_tmp = self.p_global[p_index]
            # print(p_tmp)
            p = p_tmp[:, p_index]

            # print(p)

            # print(predict_x)
            # compute jacobi
            jacobi = self.compute_new_jacobi(camera_pan=self.camera_pose[0],  camera_tilt=self.camera_pose[1],
                                             foc=self.camera_pose[2], rays=self.ray_global[index_before.astype(int)])

            print(self.ray_global[index_before.astype(int)])

            # print(jacobi.shape)

            s_k = np.dot(np.dot(jacobi, p), jacobi.T) + np.eye(2*len(index_before))

            k_k = np.dot(np.dot(p, jacobi.T), np.linalg.inv(s_k))

            print(k_k.shape)
            print(y_k.shape)
            k_mul_y = np.dot(k_k, y_k)

            print(k_mul_y)

        self.visualize_points(points_before, (255, 0, 0))
        self.visualize_points(points_next, (0,0,0))

        cv.imshow("test", self.img)
        cv.waitKey(0)


slam = PtzSlam("./two_point_calib_dataset/util/highlights_soccer_model.mat",
               "./two_point_calib_dataset/highlights/seq3_anno.mat",
               "./synthesize_data.mat")

slam.main_algorithm()
