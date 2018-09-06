"""
This is a Prototype for PTZ camera SLAM on sports applications.
2018.9
"""

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
from scipy.optimize import least_squares
from bundle_adj import BundleAdj

class PtzSlam:
    def __init__(self, model_path, annotation_path, bounding_box_path):
        """
        :param model_path: path for basketball model
        :param annotation_path: path for ground truth camera poses
        :param data_path: path for synthesized rays
        """
        self.width = 1280
        self.height = 720

        court_model = sio.loadmat(model_path)
        self.line_index = court_model['line_segment_index']
        self.points = court_model['points']

        seq = sio.loadmat(annotation_path)
        self.annotation = seq["annotation"]
        self.meta = seq['meta']


        bounding_box = sio.loadmat(bounding_box_path)
        self.bounding_box_mask_list = []
        for i in range(bounding_box['bounding_box'].shape[1]):
            tmp_mask = np.ones([self.height, self.width])
            for j in range(bounding_box['bounding_box'][0][i].shape[0]):
                if bounding_box['bounding_box'][0][i][j][4] > 0.6:
                    for x in range(int(bounding_box['bounding_box'][0][i][j][0]),
                                   int(bounding_box['bounding_box'][0][i][j][2])):
                        for y in range(int(bounding_box['bounding_box'][0][i][j][1]),
                                       int(bounding_box['bounding_box'][0][i][j][3])):
                            tmp_mask[y, x] = 0
            self.bounding_box_mask_list.append(tmp_mask)
            print("loading bounding box ... %d" % i)

        """
        initialize the fixed parameters of our algorithm
        u, v, base_rotation and c
        """
        self.u, self.v = self.annotation[0][0]['camera'][0][0:2]
        self.base_rotation = np.zeros([3, 3])
        cv.Rodrigues(self.meta[0][0]["base_rotation"][0], self.base_rotation)
        self.c = self.meta[0][0]["cc"][0]

        """parameters to be updated"""
        self.camera_pose = np.ndarray([3])
        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

        """global rays and covariance matrix"""
        self.ray_global = np.ndarray([0, 2])
        self.p_global = np.zeros([3, 3])

        """set the ground truth camera pose for whole sequence"""
        self.ground_truth_pan = np.ndarray([self.annotation.size])
        self.ground_truth_tilt = np.ndarray([self.annotation.size])
        self.ground_truth_f = np.ndarray([self.annotation.size])
        for i in range(self.annotation.size):
            self.ground_truth_pan[i], self.ground_truth_tilt[i], self.ground_truth_f[i] \
                = self.annotation[0][i]['ptz'].squeeze()

        """filter ground truth camera pose. Only for synthesized court"""
        # self.ground_truth_pan = sig.savgol_filter(self.ground_truth_pan, 181, 1)
        # self.ground_truth_tilt = sig.savgol_filter(self.ground_truth_tilt, 181, 1)
        # self.ground_truth_f = sig.savgol_filter(self.ground_truth_f, 181, 1)

        """camera pose sequence"""
        self.predict_pan = np.zeros([self.annotation.size])
        self.predict_tilt = np.zeros([self.annotation.size])
        self.predict_f = np.zeros([self.annotation.size])

        """add keyframe for bundle adjustment for our system"""

    @staticmethod
    def get_basketball_image_gray(index):
        """
        :param index: image index for basketball sequence
        :return: gray image
        """
        # img = cv.imread("./basketball/basketball/synthesize_images/" + str(index) + ".jpg")
        img = cv.imread("./basketball/basketball/images/000" + str(index + 84000) + ".jpg")
        # img = cv.imread("./two_point_calib_dataset/highlights/seq3/0" + str(index * 6 + 515) + ".jpg")
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img_gray

    @staticmethod
    def visualize_points(img, points, pt_color, rad):
        """draw some colored points in img"""
        for j in range(len(points)):
            cv.circle(img, (int(points[j][0]), int(points[j][1])), color=pt_color, radius=rad, thickness=2)

    @staticmethod
    def get_overlap_index(index1, index2):
        index1_overlap = np.ndarray([0], np.int8)
        index2_overlap = np.ndarray([0], np.int8)
        ptr1 = 0
        ptr2 = 0
        while ptr1 < len(index1) and ptr2 < len(index2):
            if index1[ptr1] == index2[ptr2]:
                index1_overlap = np.append(index1_overlap, ptr1)
                index2_overlap = np.append(index2_overlap, ptr2)
                ptr1 += 1
                ptr2 += 1
            elif index1[ptr1] < index2[ptr2]:
                ptr1 += 1
            elif index1[ptr1] > index2[ptr2]:
                ptr2 += 1
        return index1_overlap, index2_overlap

    @staticmethod
    def detect_harris_corner_grid(gray_img, row, column):
        mask = np.zeros_like(gray_img, dtype=np.uint8)

        grid_height = gray_img.shape[0] // row
        grid_width = gray_img.shape[1] // column

        all_harris = np.ndarray([0, 1, 2], dtype=np.float32)

        for i in range(row):
            for j in range(column):
                mask.fill(0)
                grid_y1 = i * grid_height
                grid_x1 = j * grid_width

                if i == row - 1:
                    grid_y2 = gray_img.shape[0]
                else:
                    grid_y2 = i * grid_height + grid_height

                if j == column - 1:
                    grid_x2 = gray_img.shape[1]
                else:
                    grid_x2 = j * grid_width + grid_width

                mask[grid_y1:grid_y2, grid_x1:grid_x2] = 1
                grid_harris = cv.goodFeaturesToTrack(gray_img, maxCorners=5,
                                                     qualityLevel=0.2, minDistance=10, mask=mask.astype(np.uint8))

                if grid_harris is not None:
                    all_harris = np.concatenate([all_harris, grid_harris], axis=0)

        return all_harris

    @staticmethod
    def detect_sift_corner(gray_img):

        sift_pts = []

        sift = cv.xfeatures2d.SIFT_create(nfeatures=50)
        kp = sift.detect(gray_img, None)

        for i in range(len(kp)):
            sift_pts.append(kp[i].pt[0])
            sift_pts.append(kp[i].pt[1])

        return np.array(sift_pts, dtype=np.float32).reshape([-1, 1, 2])

    def get_ptz(self, index):
        return np.array([self.ground_truth_pan[index], self.ground_truth_tilt[index], self.ground_truth_f[index]])

    def compute_new_jacobi(self, camera_pan, camera_tilt, foc, rays):
        """
        compute jacobi matrix
        :param camera_pan:
        :param camera_tilt:
        :param foc:
        :param rays: [RayNumber * 2]
        :return: [2 * RayNumber, 3 + 2 * RayNumber]
        """
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

    def get_observation_from_rays(self, pan, tilt, f, rays):
        """
        return all 2d points(with features),
        corresponding rays(with features) and indexes of these points IN THE IMAGE.
        :param pan:
        :param tilt:
        :param f:
        :param rays: [N, 2]
        :return: 2-d points: [n, 2] rays: [n, 2], indexes [n]
        """
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

    def get_rays_from_observation(self, pan, tilt, f, points):
        """
        get a list of rays from 2d points and camera pose
        :param pan:
        :param tilt:
        :param f:
        :param points: [PointNumber, 2]
        :return: [RayNumber(=PointNumber), 2]
        """
        rays = np.ndarray([0, 2])
        for i in range(len(points)):
            angles = TransFunction.from_2d_to_pan_tilt(self.u, self.v, f, pan, tilt, points[i][0], points[i][1])
            rays = np.row_stack([rays, angles])
        return rays

    def output_camera_error(self, now_index):
        """output the error of camera pose compared to ground truth"""
        ground_truth = self.get_ptz(now_index)
        pan, tilt, f = self.camera_pose - ground_truth
        print("%.3f %.3f, %.1f" % (pan, tilt, f), "\n")

    def draw_camera_plot(self):
        """percentage"""
        plt.figure("pan percentage error")
        x = np.array([i for i in range(slam.annotation.size)])
        # plt.plot(x, self.ground_truth_pan, 'r', label='ground truth')
        plt.plot(x, (self.predict_pan - self.ground_truth_pan) / self.ground_truth_pan * 100, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("error %")
        plt.legend(loc="best")

        plt.figure("tilt percentage error")
        x = np.array([i for i in range(slam.annotation.size)])
        # plt.plot(x, self.ground_truth_tilt, 'r', label='ground truth')
        plt.plot(x, (self.predict_tilt - self.ground_truth_tilt) / self.ground_truth_tilt * 100, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("error %")
        plt.legend(loc="best")

        plt.figure("f percentage error")
        x = np.array([i for i in range(slam.annotation.size)])
        # plt.plot(x, self.ground_truth_f, 'r', label='ground truth')
        plt.plot(x, (self.predict_f - self.ground_truth_f) / self.ground_truth_f * 100, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("error %")
        plt.legend(loc="best")

        """absolute value"""
        plt.figure("pan")
        x = np.array([i for i in range(slam.annotation.size)])
        plt.plot(x, self.ground_truth_pan, 'r', label='ground truth')
        plt.plot(x, self.predict_pan, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("pan angle")
        plt.legend(loc="best")

        plt.figure("tilt")
        x = np.array([i for i in range(slam.annotation.size)])
        plt.plot(x, self.ground_truth_tilt, 'r', label='ground truth')
        plt.plot(x, self.predict_tilt, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("tilt angle")
        plt.legend(loc="best")

        plt.figure("f")
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

    def load_camera_mat(self, path):
        camera_pos = sio.loadmat(path)
        self.predict_pan = camera_pos['predict_pan'].squeeze()
        self.predict_tilt = camera_pos['predict_tilt'].squeeze()
        self.predict_f = camera_pos['predict_f'].squeeze()

        self.ground_truth_pan = camera_pos['ground_truth_pan'].squeeze()
        self.ground_truth_tilt = camera_pos['ground_truth_tilt'].squeeze()
        self.ground_truth_f = camera_pos['ground_truth_f'].squeeze()

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

    def remove_player_feature(self, index, keypoints):
        this_mask = self.bounding_box_mask_list[index]
        ret_keypoints = np.ndarray([0, 1, 2], dtype=np.float32)
        for i in range(keypoints.shape[0]):
            x, y = int(keypoints[i][0][0]), int(keypoints[i][0][1])
            if this_mask[y, x] == 1:
                ret_keypoints = np.concatenate([ret_keypoints, (keypoints[i]).reshape([1, 1, 2])], axis=0)

        return ret_keypoints

    def init_system(self, index):
        self.camera_pose = self.get_ptz(index)
        """first frame to initialize global_rays"""
        begin_frame = self.get_basketball_image_gray(index)

        # first_frame_kp = PtzSlam.detect_harris_corner_grid(first_frame, 4, 4, first)
        begin_frame_kp = PtzSlam.detect_sift_corner(begin_frame)
        begin_frame_kp = self.remove_player_feature(index, begin_frame_kp)

        """use key points in first frame to get init rays"""
        init_rays = self.get_rays_from_observation(
            self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], begin_frame_kp.squeeze())

        """add rays in frame 1 to global rays"""
        self.ray_global = np.row_stack([self.ray_global, init_rays])

        """initialize global p using global rays"""
        self.p_global = 0.001 * np.eye(3 + 2 * len(self.ray_global))
        self.p_global[2][2] = 1

        """q_k: covariance matrix of noise for state(camera pose)"""

        previous_frame_kp = begin_frame_kp
        previous_index = np.array([i for i in range(len(self.ray_global))])

        self.predict_pan[index], self.predict_tilt[index], self.predict_f[index] = self.camera_pose

        return previous_frame_kp, previous_index

    def optical_flow_matching(self, pre_img, next_img, pre_points, pre_index):
        next_points, status, err = cv.calcOpticalFlowPyrLK(
            pre_img, next_img, pre_points, None, winSize=(31, 31))

        good_next_points = np.ndarray([0, 2])
        good_pre_points = np.ndarray([0, 2])
        good_index = np.ndarray([0])

        for j in range(len(next_points)):
            if err[j] < 20 and 0 < next_points[j][0][0] < self.width and 0 < next_points[j][0][1] < self.height:
                good_index = np.append(good_index, pre_index[j])
                good_next_points = np.row_stack([good_next_points, next_points[j][0]])
                good_pre_points = np.row_stack([good_pre_points, pre_points[j][0]])

        return good_pre_points, good_next_points, good_index

    def run_ransac(self, points1, points2, index):
        ransac_mask = np.ndarray([len(points1)])
        _, ransac_mask = cv.findHomography(srcPoints=points1, dstPoints=points2,
                                           ransacReprojThreshold=0.5, method=cv.FM_RANSAC, mask=ransac_mask)
        inner_kp = np.ndarray([0, 2])
        inner_index = np.ndarray([0])

        for j in range(len(points1)):
            if ransac_mask[j] == 1:
                inner_kp = np.row_stack([inner_kp, points2[j]])
                inner_index = np.append(inner_index, index[j])

        return inner_kp, inner_index, ransac_mask

    def ekf_update(self, i, matched_kp, next_index):
        # get 2d points, rays and indexes in all landmarks with predicted camera pose
        predict_points, predict_rays, inner_point_index = self.get_observation_from_rays(
            self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], self.ray_global)

        # compute y_k
        overlap1, overlap2 = PtzSlam.get_overlap_index(next_index, inner_point_index)
        y_k = matched_kp[overlap1] - predict_points[overlap2]
        y_k = y_k.flatten()

        matched_inner_point_index = next_index[overlap1]

        # get p matrix for this iteration from p_global
        p_index = (np.concatenate([[0, 1, 2], matched_inner_point_index + 3,
                                   matched_inner_point_index + len(matched_inner_point_index) + 3])).astype(int)

        p = self.p_global[p_index][:, p_index]

        # compute jacobi
        jacobi = self.compute_new_jacobi(camera_pan=self.camera_pose[0], camera_tilt=self.camera_pose[1],
                                         foc=self.camera_pose[2],
                                         rays=self.ray_global[matched_inner_point_index.astype(int)])
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

        print("speed", self.delta_pan, self.delta_tilt, self.delta_zoom)

        # update global rays
        for j in range(len(matched_inner_point_index)):
            self.ray_global[int(matched_inner_point_index[j])][0:2] += k_mul_y[2 * j + 3: 2 * j + 5]

        # update global p
        update_p = np.dot(np.eye(3 + 2 * len(matched_inner_point_index)) - np.dot(k_k, jacobi), p)
        self.p_global[0:3, 0:3] = update_p[0:3, 0:3]
        for j in range(len(matched_inner_point_index)):
            for k in range(len(matched_inner_point_index)):
                self.p_global[
                    3 + 2 * int(matched_inner_point_index[j]), 3 + 2 * int(matched_inner_point_index[k])] = \
                    update_p[3 + 2 * j, 3 + 2 * k]
                self.p_global[
                    3 + 2 * int(matched_inner_point_index[j]) + 1, 3 + 2 * int(matched_inner_point_index[k]) + 1] = \
                    update_p[3 + 2 * j + 1, 3 + 2 * k + 1]

    def delete_outliers(self, ransac_mask):
        """
        delete ransac outliers from global ray
        """
        delete_index = np.ndarray([0])
        for j in range(len(ransac_mask)):
            if ransac_mask[j] == 0:
                delete_index = np.append(delete_index, j)

        self.ray_global = np.delete(self.ray_global, delete_index, axis=0)

        p_delete_index = np.concatenate([delete_index + 3, delete_index + len(delete_index) + 3], axis=0)

        self.p_global = np.delete(self.p_global, p_delete_index, axis=0)
        self.p_global = np.delete(self.p_global, p_delete_index, axis=1)

    def add_new_points(self, i):
        points_update, in_rays_update, index_update = self.get_observation_from_rays(
            self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], self.ray_global)
        img_new = self.get_basketball_image_gray(i)

        """set the mask"""
        mask = np.ones(img_new.shape, np.uint8)
        for j in range(len(points_update)):
            x, y = points_update[j]
            up_bound = int(max(0, y - 50))
            low_bound = int(min(self.height, y + 50))
            left_bound = int(max(0, x - 50))
            right_bound = int(min(self.width, x + 50))
            mask[up_bound:low_bound, left_bound:right_bound] = 0

        all_new_frame_kp = PtzSlam.detect_sift_corner(img_new)
        all_new_frame_kp = self.remove_player_feature(i, all_new_frame_kp)

        new_frame_kp = np.ndarray([0, 1, 2])

        """only select those are far from previous corners"""
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

        return points_update.astype(np.float32), index_update

    def main_algorithm(self, first, step_length):
        """
        This is main function for SLAM system.
        Run this function to begin tracking and mapping
        :param first: the start frame index
        :param step_length: step length between consecutive frames
        """
        previous_frame_kp, previous_index = self.init_system(first)
        bundle_adj = BundleAdj(self)

        for i in range(first + step_length, self.annotation.size, step_length):

            print("=====The ", i, " iteration=====Total %d global rays\n" % len(self.ray_global))

            """
            ===============================
            0. feature matching step
            ===============================
            """
            pre_img = PtzSlam.get_basketball_image_gray(i - step_length)
            next_img = PtzSlam.get_basketball_image_gray(i)

            ransac_previous_kp, ransac_next_kp, ransac_index = \
                self.optical_flow_matching(pre_img, next_img, previous_frame_kp, previous_index)

            matched_kp, next_index, ransac_mask = self.run_ransac(ransac_previous_kp, ransac_next_kp, ransac_index)

            """
            ===============================
            1. predict step
            ===============================
            """
            """update camera pose with constant speed model"""
            # self.camera_pose += [self.delta_pan, self.delta_tilt, self.delta_zoom]

            """update p_global"""
            q_k = 5 * np.diag([0.001, 0.001, 1])
            self.p_global[0:3, 0:3] = self.p_global[0:3, 0:3] + q_k

            """
            ===============================
            2. update step
            ===============================
            """

            self.ekf_update(i, matched_kp, next_index)

            """
            ===============================
            3. delete outliers
            ===============================
            """
            self.delete_outliers(ransac_mask)

            """
            ===============================
            4.  add new features & update previous frame
            ===============================
            """
            previous_frame_kp, previous_index = self.add_new_points(i)

            """
            ===============================
            5. key frame and restart system
            ===============================
            """
            restart = 2600
            if i == restart:
                self.camera_pose = bundle_adj.add_key_frame(
                    restart, self.camera_pose[0], self.camera_pose[1], self.camera_pose[2])
                previous_frame_kp, previous_index = self.init_system(restart)


if __name__ == "__main__":
    # slam = PtzSlam("./two_point_calib_dataset/util/highlights_soccer_model.mat",
    #                "./two_point_calib_dataset/highlights/seq3_anno.mat",
    #                "./synthesize_data.mat",
    #                "./objects_sort_seq3.mat")

    slam = PtzSlam("./basketball/basketball_model.mat",
                   "./basketball/basketball/basketball_anno.mat",
                   "./objects_sort.mat")


    slam.main_algorithm(first=0, step_length=1)

    slam.draw_camera_plot()
    slam.save_camera_to_mat()
