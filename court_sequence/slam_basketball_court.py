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
import scipy.signal as sig


class PtzSlam:
    def __init__(self, model_path, annotation_path, data_path):

        self.width = 1280
        self.height = 720
        # self.img = np.zeros((self.height, self.width, 3), np.uint8)

        court_model = sio.loadmat(model_path)
        self.line_index = court_model['line_segment_index']
        self.points = court_model['points']

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

        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = self.meta[0][0]["cc"][0]

        # parameters to be updated
        self.camera_pose = np.ndarray([3])
        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

        self.ray_global = np.ndarray([0, 2])
        self.p_global = np.zeros([3, 3])

        self.ground_truth_pan = np.ndarray([self.annotation.size])
        self.ground_truth_tilt = np.ndarray([self.annotation.size])
        self.ground_truth_f = np.ndarray([self.annotation.size])
        for i in range(self.annotation.size):
            self.ground_truth_pan[i], self.ground_truth_tilt[i], self.ground_truth_f[i] \
                = self.annotation[0][i]['ptz'].squeeze()

        # self.ground_truth_pan = sig.savgol_filter(self.ground_truth_pan, 181, 1)
        # self.ground_truth_tilt = sig.savgol_filter(self.ground_truth_tilt, 181, 1)
        # self.ground_truth_f = sig.savgol_filter(self.ground_truth_f, 181, 1)

        # camera pose sequence
        self.predict_pan = np.ndarray([self.annotation.size])
        self.predict_tilt = np.ndarray([self.annotation.size])
        self.predict_f = np.ndarray([self.annotation.size])

    def get_ptz(self, index):
        return np.array([self.ground_truth_pan[index], self.ground_truth_tilt[index], self.ground_truth_f[index]])

    # return [CornerNumber * 1 * 2]
    @staticmethod
    def get_basketball_image_gray(index):
        # img = cv.imread("./basketball/basketball/synthesize_images/" + str(index) + ".jpg")
        img = cv.imread("./basketball/basketball/images/000" + str(index+84000) + ".jpg")
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img_gray

    @staticmethod
    def get_basketball_image_rgb(index):
        # img = cv.imread("./basketball/basketball/synthesize_images/" + str(index) + ".jpg")
        img = cv.imread("./basketball/basketball/images/000" + str(index + 84000) + ".jpg")
        return img

    """
    compute the H_jacobi matrix
    rays: [RayNumber * 2]
    return: [2 * RayNumber, 3 + 2 * RayNumber]
    """
    def compute_new_jacobi(self, camera_pan, camera_tilt, foc, rays):
        ray_num = len(rays)

        delta_angle = 0.001
        delta_f = 0.1

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
        points = np.ndarray([0, 2])
        inner_rays = np.ndarray([0, 2])
        index = np.ndarray([0])

        for j in range(len(rays)):
            tmp = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, rays[j][0], rays[j][1])
            if 0 < tmp[0] < self.width and 0 < tmp[1] < self.height:
                inner_rays = np.row_stack([inner_rays, rays[j]])
                points = np.row_stack([points, np.asarray(tmp)])
                index = np.concatenate([index, [j]], axis=0)

        return points, inner_rays, index

    """
    get a list of rays(with features) from 2d points and camera pose
    points: [PointNumber * 2]
    """
    def get_rays_from_observation(self, pan, tilt, f, points):
        rays = np.ndarray([0, 2])
        for i in range(len(points)):
            angles = TransFunction.from_2d_to_pan_tilt(self.u, self.v, f, pan, tilt, points[i][0], points[i][1])
            rays = np.row_stack([rays, angles])
        return rays

    # draw some colored points in img
    @staticmethod
    def visualize_points(img, points, pt_color):
        for j in range(len(points)):
            cv.circle(img, (int(points[j][0]), int(points[j][1])), color=pt_color, radius=8, thickness=2)

    # output the error of camera pose compared to ground truth
    def output_camera_error(self, now_index):
        ground_truth = self.get_ptz(now_index)
        pan, tilt, f = self.camera_pose - ground_truth
        print("%.3f %.3f, %.1f" % (pan, tilt, f), "\n")

    def draw_camera_plot(self):
        plt.figure(0)
        x = np.array([i for i in range(slam.annotation.size)])
        plt.plot(x, self.ground_truth_pan, 'r', label='ground truth')
        plt.plot(x, self.predict_pan, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("pan angle")
        plt.legend(loc = "best")

        plt.figure(1)
        x = np.array([i for i in range(slam.annotation.size)])
        plt.plot(x, self.ground_truth_tilt, 'r', label='ground truth')
        plt.plot(x, self.predict_tilt, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("tilt angle")
        plt.legend(loc="best")

        plt.figure(2)
        x = np.array([i for i in range(slam.annotation.size)])
        plt.plot(x, self.ground_truth_f, 'r', label='ground truth')
        plt.plot(x, self.predict_f, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("f")
        plt.legend(loc="best")
        plt.show()

    def save_camera_to_mat(self):
        camera_pose = dict()

        camera_pose['ground_truth_pan'] = self.ground_truth_pan
        camera_pose['ground_truth_tilt'] = self.ground_truth_tilt
        camera_pose['ground_truth_f'] = self.ground_truth_f

        camera_pose['predict_pan'] = self.predict_pan
        camera_pose['predict_tilt'] = self.predict_tilt
        camera_pose['predict_f'] = self.predict_f

        sio.savemat('camera_pose.mat', mdict=camera_pose)

    def draw_box(self, img, ray):
        half_edge = 0.05

        if len(ray) == 2:
            position = TransFunction.from_ray_to_relative_3d(ray[0], ray[1])
        else:
            position = TransFunction.from_3d_to_relative_3d(self.c, self.base_rotation, ray)

        pt1 = position + [half_edge, half_edge, half_edge]
        pt2 = position + [half_edge, half_edge, -half_edge]
        pt3 = position + [half_edge, -half_edge, half_edge]
        pt4 = position + [half_edge, -half_edge, -half_edge]
        pt5 = position + [-half_edge, half_edge, half_edge]
        pt6 = position + [-half_edge, half_edge, -half_edge]
        pt7 = position + [-half_edge, -half_edge, half_edge]
        pt8 = position + [-half_edge, -half_edge, -half_edge]

        center = TransFunction.from_relative_3d_to_2d(
            self.u, self.v, self.camera_pose[2], self.camera_pose[0], self.camera_pose[1], position)
        pt1_2d = TransFunction.from_relative_3d_to_2d(
            self.u, self.v, self.camera_pose[2], self.camera_pose[0], self.camera_pose[1], pt1)
        pt2_2d = TransFunction.from_relative_3d_to_2d(
            self.u, self.v, self.camera_pose[2], self.camera_pose[0], self.camera_pose[1], pt2)
        pt3_2d = TransFunction.from_relative_3d_to_2d(
            self.u, self.v, self.camera_pose[2], self.camera_pose[0], self.camera_pose[1], pt3)
        pt4_2d = TransFunction.from_relative_3d_to_2d(
            self.u, self.v, self.camera_pose[2], self.camera_pose[0], self.camera_pose[1], pt4)
        pt5_2d = TransFunction.from_relative_3d_to_2d(
            self.u, self.v, self.camera_pose[2], self.camera_pose[0], self.camera_pose[1], pt5)
        pt6_2d = TransFunction.from_relative_3d_to_2d(
            self.u, self.v, self.camera_pose[2], self.camera_pose[0], self.camera_pose[1], pt6)
        pt7_2d = TransFunction.from_relative_3d_to_2d(
            self.u, self.v, self.camera_pose[2], self.camera_pose[0], self.camera_pose[1], pt7)
        pt8_2d = TransFunction.from_relative_3d_to_2d(
            self.u, self.v, self.camera_pose[2], self.camera_pose[0], self.camera_pose[1], pt8)

        cv.line(img, (int(pt1_2d[0]), int(pt1_2d[1])), (int(pt2_2d[0]), int(pt2_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(pt1_2d[0]), int(pt1_2d[1])), (int(pt3_2d[0]), int(pt3_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(pt1_2d[0]), int(pt1_2d[1])), (int(pt5_2d[0]), int(pt5_2d[1])), (255, 128, 0), 2)

        cv.line(img, (int(pt2_2d[0]), int(pt2_2d[1])), (int(pt4_2d[0]), int(pt4_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(pt2_2d[0]), int(pt2_2d[1])), (int(pt6_2d[0]), int(pt6_2d[1])), (255, 128, 0), 2)

        cv.line(img, (int(pt3_2d[0]), int(pt3_2d[1])), (int(pt4_2d[0]), int(pt4_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(pt3_2d[0]), int(pt3_2d[1])), (int(pt7_2d[0]), int(pt7_2d[1])), (255, 128, 0), 2)

        cv.line(img, (int(pt4_2d[0]), int(pt4_2d[1])), (int(pt8_2d[0]), int(pt8_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(pt5_2d[0]), int(pt5_2d[1])), (int(pt6_2d[0]), int(pt6_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(pt5_2d[0]), int(pt5_2d[1])), (int(pt7_2d[0]), int(pt7_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(pt6_2d[0]), int(pt6_2d[1])), (int(pt8_2d[0]), int(pt8_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(pt7_2d[0]), int(pt7_2d[1])), (int(pt8_2d[0]), int(pt8_2d[1])), (255, 128, 0), 2)

        cv.line(img, (int(center[0]), int(center[1])), (int(pt1_2d[0]), int(pt1_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(center[0]), int(center[1])), (int(pt2_2d[0]), int(pt2_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(center[0]), int(center[1])), (int(pt3_2d[0]), int(pt3_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(center[0]), int(center[1])), (int(pt4_2d[0]), int(pt4_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(center[0]), int(center[1])), (int(pt5_2d[0]), int(pt5_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(center[0]), int(center[1])), (int(pt6_2d[0]), int(pt6_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(center[0]), int(center[1])), (int(pt7_2d[0]), int(pt7_2d[1])), (255, 128, 0), 2)
        cv.line(img, (int(center[0]), int(center[1])), (int(pt8_2d[0]), int(pt8_2d[1])), (255, 128, 0), 2)

    def main_algorithm(self):

        first = 0

        # first ground truth camera pose
        self.camera_pose = self.get_ptz(first)

        # first frame to initialize global_rays
        first_frame = self.get_basketball_image_gray(first)
        first_frame_kp = cv.goodFeaturesToTrack(first_frame, 30, 0.1, 10)

        # use key points in first frame to get init rays
        init_rays = self.get_rays_from_observation(
            self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], first_frame_kp.squeeze())

        # add rays in frame 1 to global rays
        self.ray_global = np.row_stack([self.ray_global, init_rays])

        # initialize global p using global rays
        self.p_global = 0.001 * np.eye(3 + 2 * len(self.ray_global))
        self.p_global[2][2] = 1

        # q_k: covariance matrix of noise for state(camera pose)
        q_k = 5 * np.diag([0.001, 0.001, 1])

        previous_frame_kp = first_frame_kp
        previous_index = np.array([i for i in range(len(self.ray_global))])

        step_length = 1
        self.predict_pan[0], self.predict_tilt[0], self.predict_f[0] = self.camera_pose

        for i in range(first + step_length, self.annotation.size, step_length):

            print("=====The ", i, " iteration=====Total %d global rays\n" % len(self.ray_global))

            """
            ===============================
            0. matching step
            ===============================
            """
            # ground truth features for next frame. In real data we do not need to compute that
            next_frame_kp, status, err = cv.calcOpticalFlowPyrLK(
                self.get_basketball_image_gray(i-step_length), self.get_basketball_image_gray(i),
                previous_frame_kp, None, winSize=(31, 31))

            # matched_kp = np.ndarray([0, 2])
            # next_index = np.ndarray([0])
            # for j in range(len(next_frame_kp)):
            #     if err[j] < 20 and 0 < next_frame_kp[j][0][0] < self.width and 0 < next_frame_kp[j][0][1] < self.height:
            #
            #         matched_kp = np.row_stack([matched_kp, next_frame_kp[j][0]])
            #         next_index = np.append(next_index, previous_index[j])

            ransac_next_kp = np.ndarray([0, 2])
            ransac_previous_kp = np.ndarray([0, 2])
            ransac_index = np.ndarray([0])

            for j in range(len(next_frame_kp)):
                if err[j] < 20 and 0 < next_frame_kp[j][0][0] < self.width and 0 < next_frame_kp[j][0][1] < self.height:
                    ransac_index = np.append(ransac_index, previous_index[j])
                    # matched_kp = np.row_stack([matched_kp, next_frame_kp[j][0]])

                    ransac_next_kp = np.row_stack([ransac_next_kp, next_frame_kp[j][0]])
                    ransac_previous_kp = np.row_stack([ransac_previous_kp, previous_frame_kp[j][0]])


            # RANSAC algorithm
            ransac_mask = np.ndarray([len(ransac_previous_kp)])
            _, ransac_mask = cv.findHomography(srcPoints=ransac_previous_kp, dstPoints=ransac_next_kp,
                                               ransacReprojThreshold=0.5, method=cv.FM_RANSAC, mask=ransac_mask)

            # print(ransac_mask)

            matched_kp = np.ndarray([0, 2])
            next_index = np.ndarray([0])

            for j in range(len(ransac_previous_kp)):
                if ransac_mask[j] == 1:
                    matched_kp = np.row_stack([matched_kp, ransac_next_kp[j]])
                    next_index = np.append(next_index, ransac_index[j])

            """
            ===============================
            1. predict step
            ===============================
            """
            # update camera pose with constant speed model
            self.camera_pose += [self.delta_pan, self.delta_tilt, self.delta_zoom]

            # update p_global
            self.p_global[0:3, 0:3] = self.p_global[0:3, 0:3] + q_k

            """
            ===============================
            2. update step
            ===============================
            """

            # get 2d points, rays and indexes in all landmarks with predicted camera pose
            predict_points, predict_rays, inner_point_index = self.get_observation_from_rays(
                self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], self.ray_global)

            # compute y_k
            y_k = np.ndarray([0])

            matched_inner_point_index = np.ndarray([0])
            ptr = 0
            for j in range(len(matched_kp)):

                while ptr < len(inner_point_index)-1 and inner_point_index[ptr] < next_index[j]:
                    ptr += 1
                if inner_point_index[ptr] > next_index[j]:
                    continue
                y_k = np.concatenate([y_k, matched_kp[j] - predict_points[ptr]], axis=0)
                matched_inner_point_index = np.concatenate([matched_inner_point_index, [next_index[j]]], axis=0)

            img1 = self.get_basketball_image_rgb(i-step_length)
            img2 = self.get_basketball_image_rgb(i)

            self.visualize_points(img1, previous_frame_kp.squeeze(), (0, 0, 0))

            # cv.imshow("img1", img1)

            # get predicted_x: combine 3 camera parameters and rays
            predict_x = self.camera_pose
            for j in range(len(matched_inner_point_index)):
                predict_x = np.concatenate([predict_x, self.ray_global[int(matched_inner_point_index[j])][0:2]], axis=0)

            # get p matrix for this iteration from p_global
            p_index = (np.concatenate([[0, 1, 2], matched_inner_point_index + 3, matched_inner_point_index + len(matched_inner_point_index) + 3])).astype(int)
            p = self.p_global[p_index][:, p_index]

            # compute jacobi
            jacobi = self.compute_new_jacobi(camera_pan=self.camera_pose[0], camera_tilt=self.camera_pose[1],
                                             foc=self.camera_pose[2], rays=self.ray_global[matched_inner_point_index.astype(int)])

            # get Kalman gain
            r_k = 2 * np.eye(2 * len(matched_inner_point_index))
            s_k = np.dot(np.dot(jacobi, p), jacobi.T) + r_k

            k_k = np.dot(np.dot(p, jacobi.T), np.linalg.inv(s_k))

            k_mul_y = np.dot(k_k, y_k)

            # output result for updating camera: before
            print("before update camera:\n")
            self.output_camera_error(i)

            # update camera pose
            self.camera_pose += k_mul_y[0:3]

            self.predict_pan[i], self.predict_tilt[i], self.predict_f[i] = self.camera_pose

            # output result for updating camera: after
            print("after update camera:\n")
            self.output_camera_error(i)

            # update speed model
            self.delta_pan, self.delta_tilt, self.delta_zoom = k_mul_y[0:3]

            # update global rays
            for j in range(len(matched_inner_point_index)):
                self.ray_global[int(matched_inner_point_index[j])][0:2] += k_mul_y[2 * j + 3: 2 * j + 5]

            # update global p
            update_p = np.dot(np.eye(3 + 2 * len(matched_inner_point_index)) - np.dot(k_k, jacobi), p)
            self.p_global[0:3, 0:3] = update_p[0:3, 0:3]
            for j in range(len(matched_inner_point_index)):
                for k in range(len(matched_inner_point_index)):
                    self.p_global[3+2 * int(matched_inner_point_index[j]), 3 + 2 * int(matched_inner_point_index[k])] = \
                        update_p[3+2*j, 3+2*k]
                    self.p_global[3 + 2 * int(matched_inner_point_index[j]) + 1,  3 + 2 * int(matched_inner_point_index[k]) + 1] = \
                        update_p[3 + 2 * j + 1, 3 + 2 * k + 1]

            """
            ===============================
            3. delete outliers
            ===============================
            """
            # delete rays which are outliers of ransac
            delete_index = np.ndarray([0])
            for j in range(len(ransac_mask)):
                if ransac_mask[j] == 0:
                    delete_index = np.append(delete_index, j)

            self.ray_global = np.delete(self.ray_global, delete_index, axis=0)

            p_delete_index = np.concatenate([delete_index + 3, delete_index + len(delete_index) + 3], axis=0)

            self.p_global = np.delete(self.p_global, p_delete_index, axis=0)
            self.p_global = np.delete(self.p_global, p_delete_index, axis=1)

            points_update, in_rays_update, index_update = self.get_observation_from_rays(
                self.camera_pose[0], self.camera_pose[1], self.camera_pose[2],  self.ray_global)


            self.visualize_points(img2, matched_kp, (0, 0, 255))

            # for j in range(len(in_rays_update)):
            #     self.draw_box(img2, in_rays_update[j])

            # self.draw_box(img2, [28.6512, 15.24, 0])

            cv.imshow("test", img2)
            cv.waitKey(0)

            """
            ===============================
            4.  add new features
            ===============================
            """
            # add new rays to the image
            img_new = self.get_basketball_image_gray(i)

            # set the mask
            mask = np.ones(img_new.shape, np.uint8)
            for j in range(len(points_update)):
                x, y = points_update[j]
                up_bound = int(max(0, y-50))
                low_bound = int(min(self.height, y+50))
                left_bound = int(max(0, x-50))
                right_bound = int(min(self.width, x+50))
                mask[up_bound:low_bound, left_bound:right_bound] = 0

            # find new harris corners for next frame
            all_new_frame_kp = cv.goodFeaturesToTrack(img_new, 30, 0.1, 10)

            new_frame_kp = np.ndarray([0, 1, 2])

            # only select those are far from previous corners
            for j in range(len(all_new_frame_kp)):
                if mask[int(all_new_frame_kp[j, 0, 1]), int(all_new_frame_kp[j, 0, 0])] == 1:
                    new_frame_kp = np.concatenate([new_frame_kp, (all_new_frame_kp[j]).reshape([1, 1, 2])], axis=0)

            points_update = points_update.reshape([points_update.shape[0], 1, 2])
            if new_frame_kp is not None:
                new_rays = self.get_rays_from_observation(
                    self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], new_frame_kp.squeeze(1))

                now_point_num = len(self.ray_global)

                for j in range(len(new_rays)):
                    self.ray_global = np.row_stack([self.ray_global, new_rays[j]])
                    self.p_global = np.row_stack([self.p_global, np.zeros([2, self.p_global.shape[1]])])
                    self.p_global = np.column_stack([self.p_global, np.zeros([self.p_global.shape[0], 2])])
                    self.p_global[self.p_global.shape[0] - 1, self.p_global.shape[1] - 1] = 0.01

                    index_update = np.concatenate([index_update, [now_point_num + j]], axis=0)

                points_update = np.concatenate([points_update, new_frame_kp], axis=0)

            previous_index = index_update
            previous_frame_kp = points_update.astype(np.float32)


if __name__ == "__main__":

    slam = PtzSlam("./basketball/basketball_model.mat",
                   "./basketball/basketball/basketball_anno.mat",
                   "./synthesize_data.mat")

    slam.main_algorithm()

    # camera_pos = sio.loadmat("./camera_pose.mat")
    # slam.predict_pan = camera_pos['predict_pan'].squeeze()
    # slam.predict_tilt = camera_pos['predict_tilt'].squeeze()
    # slam.predict_f = camera_pos['predict_f'].squeeze()
    #
    # slam.ground_truth_pan = camera_pos['ground_truth_pan'].squeeze()
    # slam.ground_truth_tilt = camera_pos['ground_truth_tilt'].squeeze()
    # slam.ground_truth_f = camera_pos['ground_truth_f'].squeeze()

    slam.draw_camera_plot()
    slam.save_camera_to_mat()
