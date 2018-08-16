"""
PTZ camera SLAM tested on synthesized data
2018.8
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
import statistics
from sklearn.preprocessing import normalize
from math import *
from transformation import TransFunction


class PtzSlam:
    def __init__(self, model_path, annotation_path, data_path):

        self.width = 1280
        self.height = 720
        self.img = np.zeros((self.height, self.width, 3), np.uint8)

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

        # this is synthesized rays to generate 2d-point. Real data does not have this variable
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

    # given image index, return 2d points and their features with noises
    def get_image(self, index):
        pan, tilt, f = self.get_ptz(index)
        points = np.ndarray([0, self.all_rays.shape[1]])

        for i in range(len(self.all_rays)):
            x, y = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, self.all_rays[i][0],
                                                     self.all_rays[i][1])
            x += random.uniform(-5, 5)
            y += random.uniform(-5, 5)

            if 0 < x < self.width and 0 < y < self.height:
                points = np.row_stack((points, np.concatenate([np.asarray([x, y]), self.all_rays[i][2:18]], axis=0)))

        return points

    # compute the H_jacobi matrix
    def compute_new_jacobi(self, camera_pan, camera_tilt, foc, rays):
        ray_num = len(rays)

        delta_angle = 0.001
        delta_f = 1.0

        jacobi_h = np.ndarray([2 * ray_num, 3 + 2 * ray_num])

        for i in range(ray_num):
            x_delta_pan1, y_delta_pan1 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc, camera_pan - delta_angle, camera_tilt, rays[i][0], rays[i][1])

            x_delta_pan2, y_delta_pan2 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc, camera_pan + delta_angle, camera_tilt, rays[i][0], rays[i][1])

            x_delta_tilt1, y_delta_tilt1 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc, camera_pan, camera_tilt - delta_angle, rays[i][0], rays[i][1])

            x_delta_tilt2, y_delta_tilt2 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc, camera_pan, camera_tilt + delta_angle, rays[i][0], rays[i][1])

            x_delta_f1, y_delta_f1 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc - delta_f, camera_pan, camera_tilt, rays[i][0], rays[i][1])

            x_delta_f2, y_delta_f2 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc + delta_f, camera_pan, camera_tilt, rays[i][0], rays[i][1])

            x_delta_theta1, y_delta_theta1 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc, camera_pan, camera_tilt, rays[i][0] - delta_angle, rays[i][1])

            x_delta_theta2, y_delta_theta2 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc, camera_pan, camera_tilt, rays[i][0] + delta_angle, rays[i][1])

            x_delta_phi1, y_delta_phi1 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc, camera_pan, camera_tilt, rays[i][0], rays[i][1] - delta_angle)
            x_delta_phi2, y_delta_phi2 = TransFunction.from_pan_tilt_to_2d(
                self.u, self.v, foc, camera_pan, camera_tilt, rays[i][0], rays[i][1] + delta_angle)

            jacobi_h[2 * i][0] = (x_delta_pan2 - x_delta_pan1) / (2 * delta_angle)
            jacobi_h[2 * i][1] = (x_delta_tilt2 - x_delta_tilt1) / (2 * delta_angle)
            jacobi_h[2 * i][2] = (x_delta_f2 - x_delta_f1) / (2 * delta_f)

            jacobi_h[2 * i + 1][0] = (y_delta_pan2 - y_delta_pan1) / (2 * delta_angle)
            jacobi_h[2 * i + 1][1] = (y_delta_tilt2 - y_delta_tilt1) / (2 * delta_angle)
            jacobi_h[2 * i + 1][2] = (y_delta_f2 - y_delta_f1) / (2 * delta_f)

            for j in range(ray_num):
                if j == i:
                    jacobi_h[2 * i][3 + 2 * j] = (x_delta_theta2 - x_delta_theta1) / (2 * delta_angle)
                    jacobi_h[2 * i][3 + 2 * j + 1] = (x_delta_phi2 - x_delta_phi1) / (2 * delta_angle)

                    jacobi_h[2 * i + 1][3 + 2 * j] = (y_delta_theta2 - y_delta_theta1) / (2 * delta_angle)
                    jacobi_h[2 * i + 1][3 + 2 * j + 1] = (y_delta_phi2 - y_delta_phi1) / (2 * delta_angle)
                else:
                    jacobi_h[2 * i][3 + 2 * j] = jacobi_h[2 * i][3 + 2 * j + 1] = \
                        jacobi_h[2 * i + 1][3 + 2 * j] = jacobi_h[2 * i + 1][3 + 2 * j + 1] = 0

        return jacobi_h

    # return all 2d points(with features), corresponding rays(with features) and indexes of these points IN THE IMAGE.
    def get_observation_from_rays(self, pan, tilt, f, rays):
        points = np.ndarray([0, rays.shape[1]])
        inner_rays = np.ndarray([0, rays.shape[1]])
        index = np.ndarray([0])

        for j in range(len(rays)):
            tmp = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, rays[j][0], rays[j][1])
            if 0 < tmp[0] < self.width and 0 < tmp[1] < self.height:
                inner_rays = np.row_stack((inner_rays, rays[j]))
                points = np.row_stack((points, np.concatenate([np.asarray(tmp), rays[j][2:18]], axis=0)))
                index = np.concatenate((index, [j]), axis=0)
        return points, inner_rays, index

    # get a list of rays(with features) from 2d points and camera pose
    def get_rays_from_observation(self, pan, tilt, f, points):
        rays = np.ndarray([0, points.shape[1]])
        for i in range(len(points)):
            angles = TransFunction.from_2d_to_pan_tilt(self.u, self.v, f, pan, tilt, points[i][0], points[i][1])
            rays = np.row_stack((rays, np.concatenate([angles, points[i][2:18]], axis=0)))
        return rays

    # draw some colored points in img
    def visualize_points(self, points, pt_color):
        for j in range(len(points)):
            cv.circle(self.img, (int(points[j][0]), int(points[j][1])), color=pt_color, radius=8, thickness=2)

    def main_algorithm(self):

        random.seed(1)

        self.camera_pose = self.get_ptz(0)
        first_frame = self.get_image(0)

        init_rays = self.get_rays_from_observation(self.camera_pose[0], self.camera_pose[1], self.camera_pose[2],
                                                   first_frame)

        # add rays in frame 1 to global rays
        self.ray_global = np.concatenate([self.ray_global, init_rays], axis=0)

        # initialize global p using global rays
        self.p_global = 0.01 * np.eye(3 + 2 * len(self.ray_global))
        self.p_global[2][2] = 1

        q_k = 1 * np.diag([0.01, 0.01, 1])

        for i in range(1, 30):
            self.img.fill(255)

            # ground truth features for next frame. In real data we do not need to compute that
            next_frame = self.get_image(i)

            # add the camera pose with constant speed model
            self.camera_pose += [self.delta_pan, self.delta_tilt, self.delta_zoom]

            # get 2d points, rays and indexes in all landmarks with predicted camera pose
            predict_points, predict_rays, inner_point_index = self.get_observation_from_rays(
                self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], self.ray_global)

            # update p_global
            self.p_global[0:3, 0:3] = self.p_global[0:3, 0:3] + q_k

            y_k = np.ndarray([0])
            matched_inner_point_index = np.ndarray([0])

            for j in range(len(predict_points)):
                for k in range(len(next_frame)):
                    if np.linalg.norm(next_frame[k][2:18] - predict_points[j][2:18]) < 0.01:
                        y_k = np.concatenate((y_k, next_frame[k][0:2] - predict_points[j][0:2]), axis=0)

                        matched_inner_point_index = np.concatenate((matched_inner_point_index, [inner_point_index[j]]), axis=0)

            print("matched index", matched_inner_point_index)



            predict_x = self.camera_pose
            for j in range(len(matched_inner_point_index)):
                predict_x = np.concatenate([predict_x, self.ray_global[int(matched_inner_point_index[j])][0:2]], axis=0)

            # get p matrix for this iteration from p_global
            p_index = (np.concatenate([[0, 1, 2], matched_inner_point_index + 3, matched_inner_point_index + len(matched_inner_point_index) + 3])).astype(int)

            p = self.p_global[p_index][:, p_index]

            # compute jacobi
            jacobi = self.compute_new_jacobi(camera_pan=self.camera_pose[0], camera_tilt=self.camera_pose[1],
                                             foc=self.camera_pose[2], rays=self.ray_global[matched_inner_point_index.astype(int)])

            s_k = np.dot(np.dot(jacobi, p), jacobi.T) + np.eye(2 * len(matched_inner_point_index))

            k_k = np.dot(np.dot(p, jacobi.T), np.linalg.inv(s_k))

            k_mul_y = np.dot(k_k, y_k)

            self.camera_pose += k_mul_y[0:3]

            # update speed model
            self.delta_pan, self.delta_tilt, self.delta_zoom = k_mul_y[0:3]

            # initialize new landmarks

            print("next point", len(next_frame ))

            for j in range(len(next_frame)):
                has_point = False
                for k in range(len(self.ray_global)):
                    if np.linalg.norm(next_frame[j][2:18] - self.ray_global[k][2:18]) < 0.01:
                        has_point = True

                if not has_point:
                    new_ray = self.get_rays_from_observation(
                        self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], next_frame[j:j + 1, :])

                    self.ray_global = np.concatenate([self.ray_global, new_ray], axis=0)

                    self.p_global = np.row_stack((self.p_global, np.zeros([2, self.p_global.shape[1]])))
                    self.p_global = np.column_stack((self.p_global, np.zeros([self.p_global.shape[0], 2])))
                    self.p_global[self.p_global.shape[0] - 1, self.p_global.shape[1] - 1] = 0.01

            # print("new?", self.p_global)

            print("before update rays:\n")
            theta_list = []
            phi_list = []
            for j in range(len(self.ray_global)):
                for k in range(len(self.all_rays)):
                    if np.linalg.norm(self.ray_global[j][2:18] - self.all_rays[k][2:18]) < 0.01:
                        tmp = self.ray_global[j][0:2] - self.all_rays[k][0:2]
                        theta_list.append(tmp[0])
                        phi_list.append(tmp[1])
                        # print(self.ray_global[j][0:2] - self.all_rays[k][0:2])
                        break

            print("mean", np.mean(theta_list), "sdev", statistics.stdev(theta_list))
            print("mean", np.mean(phi_list), "sdev", statistics.stdev(phi_list), "\n")

            for j in range(len(matched_inner_point_index)):
                self.ray_global[int(matched_inner_point_index[j])][0:2] += k_mul_y[2 * j + 3: 2 * j + 5]

            print("after update rays:\n")
            theta_list = []
            phi_list = []
            for j in range(len(self.ray_global)):
                for k in range(len(self.all_rays)):
                    if np.linalg.norm(self.ray_global[j][2:18] - self.all_rays[k][2:18]) < 0.01:
                        tmp = self.ray_global[j][0:2] - self.all_rays[k][0:2]
                        theta_list.append(tmp[0])
                        phi_list.append(tmp[1])
                        # print(self.ray_global[j][0:2] - self.all_rays[k][0:2])
                        break

            print("mean", np.mean(theta_list), "sdev", statistics.stdev(theta_list))
            print("mean", np.mean(phi_list), "sdev", statistics.stdev(phi_list), "\n")

            update_p = np.dot(np.eye(3 + 2 * len(matched_inner_point_index)) - np.dot(k_k, jacobi), p)
            self.p_global[0:3, 0:3] = update_p[0:3, 0:3]
            for j in range(len(matched_inner_point_index)):
                for k in range(len(matched_inner_point_index)):
                    self.p_global[3+2 * int(matched_inner_point_index[j]), 3 +2 * int(matched_inner_point_index[k])] = \
                        update_p[3+2*j, 3+2*k]
                    self.p_global[3 + 2 * int(matched_inner_point_index[j]) + 1,  3 + 2 * int(matched_inner_point_index[k]) + 1] = \
                        update_p[3 + 2 * j + 1, 3 + 2 * k + 1]

            points_update, in_rays_update, index_update = self.get_observation_from_rays(
                self.camera_pose[0], self.camera_pose[1], self.camera_pose[2],  self.ray_global)

            self.visualize_points(predict_points, (255, 0, 0))
            self.visualize_points(next_frame, (0, 0, 0))
            self.visualize_points(points_update, (0, 0, 255))

            cv.imshow("test", self.img)
            cv.waitKey(0)




slam = PtzSlam("./two_point_calib_dataset/util/highlights_soccer_model.mat",
               "./two_point_calib_dataset/highlights/seq3_anno.mat",
               "./synthesize_data.mat")

slam.main_algorithm()
