import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import random
import cv2 as cv
import statistics
import scipy.signal as sig
from sklearn.preprocessing import normalize
from math import *
from transformation import TransFunction
from image_process import *
from scipy.optimize import least_squares


class BundleAdj:
    def __init__(self, ptzslam_obj):
        self.key_frame_global_ray = np.ndarray([0, 2])
        self.key_frame_global_ray_des = np.ndarray([0, 128], dtype=np.float32)
        self.key_frame_camera = np.ndarray([0, 3])

        self.key_frame_ray_index = []
        self.key_frame_sift = []

        self.ground_truth_pan = ptzslam_obj.ground_truth_pan
        self.ground_truth_tilt = ptzslam_obj.ground_truth_tilt
        self.ground_truth_f = ptzslam_obj.ground_truth_f
        self.u = ptzslam_obj.u
        self.v = ptzslam_obj.v

        self.annotation = ptzslam_obj.annotation
        self.image_path = ptzslam_obj.image_path
        """initialize key frame map using first frame"""
        self.feature_num = 100

        first_camera = np.array([self.ground_truth_pan[0], self.ground_truth_tilt[0], self.ground_truth_f[0]])
        self.key_frame_camera = np.row_stack([self.key_frame_camera, first_camera])

        kp_init, des_init = detect_compute_sift(self.get_basketball_image_gray(0), self.feature_num)
        self.key_frame_sift.append((kp_init, des_init))

        ray_index = np.ndarray([self.feature_num])
        for i in range(len(kp_init)):
            theta, phi = TransFunction.from_2d_to_pan_tilt(
                self.u, self.v, first_camera[2], first_camera[0], first_camera[1], kp_init[i].pt[0], kp_init[i].pt[1])
            self.key_frame_global_ray = np.row_stack([self.key_frame_global_ray, [theta, phi]])
            self.key_frame_global_ray_des = np.row_stack([self.key_frame_global_ray_des, des_init[i]])
            ray_index[i] = i

        self.key_frame_ray_index.append(ray_index)

    def get_basketball_image_gray(self, index):
        """
        :param index: image index for basketball sequence
        :return: gray image
        """
        img = cv.imread(self.image_path + self.annotation[0][index]['image_name'][0])
        # img = cv.imread(self.image_path + "00000" + str(index + 515) + ".jpg")

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img_gray

    def get_observation_from_rays(self, pan, tilt, f, rays, ray_index):
        """
        :param pan:
        :param tilt:
        :param f:
        :param rays: all rays
        :param ray_index: inner index for all rays
        :return: 2d point for these rays
        """
        points = np.ndarray([0, 2])
        for j in range(len(ray_index)):
            theta = rays[int(ray_index[j])][0]
            phi = rays[int(ray_index[j])][1]
            tmp = TransFunction.from_pan_tilt_to_2d(self.u, self.v, f, pan, tilt, theta, phi)
            points = np.row_stack([points, np.asarray(tmp)])
        return points

    def fun(self, params, n_cameras, n_points):
        """
        :param params: contains camera parameters and rays
        :param n_cameras: number of camera poses
        :param n_points: number of rays
        :return: 1d residual
        """
        camera_params = params[:n_cameras * 3].reshape((n_cameras, 3))
        points_3d = params[n_cameras * 3:].reshape((n_points, 2))

        residual = np.ndarray([0])

        for i in range(n_cameras):
            kp, des = self.key_frame_sift[i]
            point_2d = np.ndarray([0])
            for j in range(len(kp)):
                point_2d = np.append(point_2d, kp[j].pt[0])
                point_2d = np.append(point_2d, kp[j].pt[1])
            if i == 0:
                proj_point = self.get_observation_from_rays(
                    self.ground_truth_pan[0], self.ground_truth_tilt[0], self.ground_truth_f[0],
                    points_3d, self.key_frame_ray_index[i])
            else:
                # frame_idx = self.key_frame[i]
                proj_point = self.get_observation_from_rays(
                    camera_params[i, 0], camera_params[i, 1], camera_params[i, 2],
                    points_3d, self.key_frame_ray_index[i])

            residual = np.append(residual, proj_point.flatten() - point_2d.flatten())

        return residual

    def add_key_frame(self, frame_index, pan, tilt, f):
        next_camera = np.array([pan, tilt, f])
        self.key_frame_camera = np.row_stack([self.key_frame_camera, next_camera])

        kp_n, des_n = detect_compute_sift(self.get_basketball_image_gray(frame_index), self.feature_num)
        self.key_frame_sift.append((kp_n, des_n))

        ray_index = np.ndarray([self.feature_num])

        inliers_max = 0
        best_key_frame = 0
        best_outliers = []
        best_inliers = []
        for i in range(self.key_frame_camera.shape[0] - 1):
            bf = cv.BFMatcher()
            matches = bf.knnMatch(des_n, self.key_frame_sift[i][1], k=2)

            """apply ratio test"""
            ratio_outliers = []
            ratio_inliers = []
            for m, n in matches:
                if m.distance > 0.7 * n.distance:
                    ratio_outliers.append(m)
                else:
                    ratio_inliers.append(m)
                    # ray_index[m.queryIdx] = m.trainIdx

            if len(ratio_inliers) > inliers_max:
                inliers_max = len(ratio_inliers)
                best_key_frame = i
                best_outliers, best_inliers = ratio_outliers, ratio_inliers

        print("best key", best_key_frame)

        ransac_previous_kp = np.ndarray([0, 2])
        ransac_next_kp = np.ndarray([0, 2])
        kp_pre = self.key_frame_sift[best_key_frame][0]

        for j in range(len(best_inliers)):
            next_idx = best_inliers[j].queryIdx
            pre_idx = best_inliers[j].trainIdx
            ransac_next_kp = np.row_stack([ransac_next_kp, [kp_n[next_idx].pt[0], kp_n[next_idx].pt[1]]])
            ransac_previous_kp = np.row_stack([ransac_previous_kp, [kp_pre[pre_idx].pt[0], kp_pre[pre_idx].pt[1]]])

        """ RANSAC algorithm"""
        ransac_mask = np.ndarray([len(ransac_previous_kp)])
        _, ransac_mask = cv.findHomography(srcPoints=ransac_next_kp, dstPoints=ransac_previous_kp,
                                           ransacReprojThreshold=0.5, method=cv.FM_RANSAC, mask=ransac_mask)
        ransac_inliers = []
        for j in range(len(ransac_previous_kp)):
            if ransac_mask[j] == 1:
                ransac_inliers.append(best_inliers[j])
                ray_index[best_inliers[j].queryIdx] = self.key_frame_ray_index[best_key_frame][best_inliers[j].trainIdx]
            else:
                best_outliers.append(best_inliers[j])

        """add new keypoints"""
        for j in range(len(best_outliers)):
            kp = kp_n[best_outliers[j].queryIdx]
            theta, phi = TransFunction.from_2d_to_pan_tilt(
                self.u, self.v, next_camera[2], next_camera[0], next_camera[1], kp.pt[0], kp.pt[1])
            self.key_frame_global_ray = np.row_stack([self.key_frame_global_ray, [theta, phi]])
            self.key_frame_global_ray_des = np.row_stack(
                [self.key_frame_global_ray_des, des_n[best_outliers[j].queryIdx]])

            ray_index[best_outliers[j].queryIdx] = self.key_frame_global_ray.shape[0] - 1

        self.key_frame_ray_index.append(ray_index)

        before_optimize = np.append(self.key_frame_camera.flatten(), self.key_frame_global_ray.flatten())

        after_optimize = least_squares(self.fun, before_optimize, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                                       args=(self.key_frame_camera.shape[0], self.key_frame_global_ray.shape[0]))

        self.key_frame_camera = \
            (after_optimize.x[:3 * self.key_frame_camera.shape[0]]).reshape([-1, 3])

        self.key_frame_global_ray = (after_optimize.x[3 * self.key_frame_camera.shape[0]:]).reshape([-1, 2])

        self.key_frame_camera[0] = [self.ground_truth_pan[0], self.ground_truth_tilt[0], self.ground_truth_f[0]]

        # img3 = cv.drawMatches(
        #     self.get_basketball_image_gray(2550), kp_n,
        #     self.get_basketball_image_gray(2560), kp_pre, ransac_inliers, None, flags=2)
        # cv.imshow("test", img3)
        # cv.waitKey(0)

        return self.key_frame_camera[self.key_frame_camera.shape[0] - 1]


