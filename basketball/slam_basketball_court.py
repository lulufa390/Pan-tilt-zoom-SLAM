"""
Main part of our system. Ray landmarks based PTZ SLAM.

Created by Luke, 2018.9
"""

import numpy as np
import scipy.io as sio
import random
import cv2 as cv
import scipy.signal as sig
import time
import copy
from sklearn.preprocessing import normalize
from math import *
from transformation import TransFunction
from scipy.optimize import least_squares
from image_process import *
from sequence_manager import SequenceManager
from relocalization import relocalization_camera
from scene_map import Map
from bundle_adjustment import bundle_adjustment
from key_frame import KeyFrame
from util import *


class PtzSlam:
    def __init__(self):
        """
        :param annotation_path: path for annotation file
        :param bounding_box_path: path for player bounding box mat file
        :param image_path: path for image folder
        """

        """parameters to be updated"""
        # self.camera_pose = np.ndarray([3])
        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

        """global rays and covariance matrix"""
        self.ray_global = np.ndarray([0, 2])
        self.p_global = np.zeros([3, 3])

        """camera pose sequence"""
        # self.predict_pan = []
        # self.predict_tilt = []
        # self.predict_f = []

        # save the image, keypoints and keypoints index in ray_global of previous frame
        self.previous_img = None
        self.previous_frame_kp = None
        self.previous_index = None

        # Map
        self.keyframe_map = Map('sift')

        self.camera = []

    def compute_H(self, pan, tilt, focal_length, rays):
        """
        This function computes the jacobian matrix H for h(x).
        h(x) is the function from predicted state(camera pose and ray landmarks) to predicted observations.
        H helps to compute Kalman gain for the EKF

        :param pan: pan angle of predicted camera pose
        :param tilt: tilt angle of predicted camera pose
        :param focal_length: focal length of predicted camera pose
        :param rays: predicted ray landmarks, [RayNumber * 2]
        :return: Jacobian matrix H, [2 * RayNumber, 3 + 2 * RayNumber]
        """
        ray_num = len(rays)

        delta_angle = 0.001
        delta_f = 0.1

        jacobi_h = np.zeros([2 * ray_num, 3 + 2 * ray_num])

        camera = copy.deepcopy(self.camera[0])

        """use approximate method to compute partial derivative."""
        for i in range(ray_num):
            camera.set_ptz([pan - delta_angle, tilt, focal_length])
            x_delta_pan1, y_delta_pan1 = camera.project_ray(rays[i])

            camera.set_ptz([pan + delta_angle, tilt, focal_length])
            x_delta_pan2, y_delta_pan2 = camera.project_ray(rays[i])

            camera.set_ptz([pan, tilt - delta_angle, focal_length])
            x_delta_tilt1, y_delta_tilt1 = camera.project_ray(rays[i])

            camera.set_ptz([pan, tilt + delta_angle, focal_length])
            x_delta_tilt2, y_delta_tilt2 = camera.project_ray(rays[i])

            camera.set_ptz([pan, tilt, focal_length - delta_f])
            x_delta_f1, y_delta_f1 = camera.project_ray(rays[i])

            camera.set_ptz([pan, tilt, focal_length + delta_f])
            x_delta_f2, y_delta_f2 = camera.project_ray(rays[i])

            camera.set_ptz([pan, tilt, focal_length])
            x_delta_theta1, y_delta_theta1 = camera.project_ray([rays[i, 0] - delta_angle, rays[i, 1]])
            x_delta_theta2, y_delta_theta2 = camera.project_ray([rays[i, 0] + delta_angle, rays[i, 1]])
            x_delta_phi1, y_delta_phi1 = camera.project_ray([rays[i, 0], rays[i, 1] - delta_angle])
            x_delta_phi2, y_delta_phi2 = camera.project_ray([rays[i, 0], rays[i, 1] + delta_angle])

            jacobi_h[2 * i][0] = (x_delta_pan2 - x_delta_pan1) / (2 * delta_angle)
            jacobi_h[2 * i][1] = (x_delta_tilt2 - x_delta_tilt1) / (2 * delta_angle)
            jacobi_h[2 * i][2] = (x_delta_f2 - x_delta_f1) / (2 * delta_f)

            jacobi_h[2 * i + 1][0] = (y_delta_pan2 - y_delta_pan1) / (2 * delta_angle)
            jacobi_h[2 * i + 1][1] = (y_delta_tilt2 - y_delta_tilt1) / (2 * delta_angle)
            jacobi_h[2 * i + 1][2] = (y_delta_f2 - y_delta_f1) / (2 * delta_f)

            for j in range(ray_num):
                """only j == i, the element of H is not zero.
                the partial derivative of one 2D point to a different landmark is always zero."""
                if j == i:
                    jacobi_h[2 * i][3 + 2 * j] = (x_delta_theta2 - x_delta_theta1) / (2 * delta_angle)
                    jacobi_h[2 * i][3 + 2 * j + 1] = (x_delta_phi2 - x_delta_phi1) / (2 * delta_angle)

                    jacobi_h[2 * i + 1][3 + 2 * j] = (y_delta_theta2 - y_delta_theta1) / (2 * delta_angle)
                    jacobi_h[2 * i + 1][3 + 2 * j + 1] = (y_delta_phi2 - y_delta_phi1) / (2 * delta_angle)

        return jacobi_h

    def init_system(self, first_img, first_camera, first_bounding_box=None):
        """
        This function initializes tracking component.
        It is called: 1. At the first frame. 2. after relocalization
        :param index: begin frame index
        :return: [N, 2] array keypoints, [N] array index in global ray
        """

        """first frame to initialize global_rays"""

        first_img_kp = detect_sift(first_img, 500)
        # first_img_kp = detect_orb(first_img, 300)

        """remove keypoints on players"""
        if first_bounding_box is not None:
            first_img_kp = first_img_kp[
                remove_player_feature(first_img_kp, first_bounding_box)]

        """use key points in first frame to get init rays"""
        init_rays = first_camera.back_project_to_rays(first_img_kp)

        """initialize rays"""
        self.ray_global = np.ndarray([0, 2])

        """add rays in frame 1 to global rays"""
        self.ray_global = np.row_stack([self.ray_global, init_rays])

        """initialize global p using global rays"""
        # @ not every sure about the 0.001 and 1
        self.p_global = 0.001 * np.eye(3 + 2 * len(self.ray_global))
        self.p_global[2][2] = 1

        """q_k: covariance matrix of noise for state(camera pose)"""

        self.previous_img = first_img
        self.previous_frame_kp = first_img_kp
        self.previous_index = np.array([i for i in range(len(self.ray_global))])

        self.camera.append(first_camera)

        # self.predict_pan.append(first_camera.pan)
        # self.predict_tilt.append(first_camera.tilt)
        # self.predict_f.append(first_camera.focal_length)

    def ekf_update(self, matched_kp, next_index, height, width):
        """
        This function update global rays and covariance matrix.

        @This function is important. Please add Math and add note for variables
        @ for example: y_k, dimension, y_k is xxxx in the equation xxx
        :param i: index for frame
        :param matched_kp: matched keypoint in that frame
        :param next_index: matched keypoint index in global ray
        """

        # get 2d points, rays and indexes in all landmarks with predicted camera pose
        # @todo what is inner_point_index?

        predict_points, inner_point_index = self.camera[-1].project_rays(
            self.ray_global, height, width)


        # compute y_k
        overlap1, overlap2 = get_overlap_index(next_index, inner_point_index)
        y_k = matched_kp[overlap1] - predict_points[overlap2]
        y_k = y_k.flatten()

        matched_inner_point_index = next_index[overlap1]

        # what is p_index?
        p_index = np.array([0, 1, 2])
        for j in range(len(matched_inner_point_index)):
            p_index = np.append(p_index, np.array([2 * matched_inner_point_index[j] + 3,
                                                   2 * matched_inner_point_index[j] + 4]))
        p_index = p_index.astype(np.int32)

        # print(p_index)

        p = self.p_global[p_index][:, p_index]

        # compute jacobi
        # is this the H_k in the EKF wikipedia?
        jacobi = self.compute_H(pan=self.camera[-1].pan, tilt=self.camera[-1].tilt,
                                focal_length=self.camera[-1].focal_length,
                                rays=self.ray_global[matched_inner_point_index.astype(int)])
        # get Kalman gain
        r_k = 2 * np.eye(2 * len(matched_inner_point_index))
        s_k = np.dot(np.dot(jacobi, p), jacobi.T) + r_k

        k_k = np.dot(np.dot(p, jacobi.T), np.linalg.inv(s_k))

        k_mul_y = np.dot(k_k, y_k)

        # update camera pose
        self.camera[-1].pan += k_mul_y[0]
        self.camera[-1].tilt += k_mul_y[1]
        self.camera[-1].focal_length += k_mul_y[2]

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
                self.p_global[
                    3 + 2 * int(matched_inner_point_index[j]), 3 + 2 * int(matched_inner_point_index[k])] = \
                    update_p[3 + 2 * j, 3 + 2 * k]
                self.p_global[
                    3 + 2 * int(matched_inner_point_index[j]) + 1, 3 + 2 * int(matched_inner_point_index[k]) + 1] = \
                    update_p[3 + 2 * j + 1, 3 + 2 * k + 1]

    def delete_outliers(self, ransac_mask):
        """
        remove_rays
        @ is it remove outliers in the global ray?
        @ for example, a ray that is located in the player body?
        delete ransac outliers from global ray
        The ray is initialized by keypoint detection in the first frame.
        In the next frame, some of the keypoints are corrected matched as inliers,
        others are outliers. The outlier is associated with a ray, that ray will be removed
        Note the ray is different from the ray in the Map().
        :param ransac_mask: 0 for ourliers, 1 for inliers
        """

        # delete global ray
        delete_index = np.ndarray([0])
        for j in range(len(ransac_mask)):
            if ransac_mask[j] == 0:
                delete_index = np.append(delete_index, j)

        self.ray_global = np.delete(self.ray_global, delete_index, axis=0)

        # delete covariance matrix
        p_delete_index = np.ndarray([0])
        for i in range(len(delete_index)):
            p_delete_index = np.append(p_delete_index, np.array([2 * delete_index[i] + 3,
                                                                 2 * delete_index[i] + 4]))

        self.p_global = np.delete(self.p_global, p_delete_index, axis=0)
        self.p_global = np.delete(self.p_global, p_delete_index, axis=1)

    def add_new_points(self, img_new, height, width, bounding_box):
        """
        add_rays
        @ is it add new rays (landmarks)?
        @ there are some names that are similar
        @ point: generally means 2D image point (or keypoint)
        @ 3D point:
        @ landmark: can be 3D point or 2-dimenstional ray


        Detect new keypoints in the current frame and add associated rays.
        In each frame, a number of keypoints are detected. These keypoints will
        be associated with new rays (given the camera pose). These new rays are
        added to the global ray to maintain the number of visible rays in the image.
        Otherwise, the number of rays will drop.
        :param i: frame index
        :return: previous keypoints and indexes
        """

        points_update, index_update = self.camera[-1].project_rays(
            self.ray_global, height, width)


        """set the mask"""
        mask = np.ones(img_new.shape[0:2], np.uint8)
        for j in range(len(points_update)):
            x, y = points_update[j]
            up_bound = int(max(0, y - 50))
            low_bound = int(min(height, y + 50))
            left_bound = int(max(0, x - 50))
            right_bound = int(min(width, x + 50))
            mask[up_bound:low_bound, left_bound:right_bound] = 0

        all_new_frame_kp = detect_sift(img_new, 500)
        # all_new_frame_kp = detect_orb(img_new, 300)

        if bounding_box is not None:
            all_new_frame_kp = all_new_frame_kp[
                remove_player_feature(all_new_frame_kp, bounding_box)]

        new_frame_kp = np.ndarray([0, 2])
        """use mask to remove feature points near existing points"""
        for j in range(len(all_new_frame_kp)):
            if mask[int(all_new_frame_kp[j, 1]), int(all_new_frame_kp[j, 0])] == 1:
                new_frame_kp = np.concatenate([new_frame_kp, (all_new_frame_kp[j]).reshape([1, 2])], axis=0)

        """if existing new points"""
        if new_frame_kp is not None:
            new_rays = self.camera[-1].back_project_to_rays(new_frame_kp)
            now_point_num = len(self.ray_global)

            """add to global ray and covariance matrix"""
            for j in range(len(new_rays)):
                self.ray_global = np.row_stack([self.ray_global, new_rays[j]])
                self.p_global = np.row_stack([self.p_global, np.zeros([2, self.p_global.shape[1]])])
                self.p_global = np.column_stack([self.p_global, np.zeros([self.p_global.shape[0], 2])])
                self.p_global[self.p_global.shape[0] - 2, self.p_global.shape[1] - 2] = 0.01
                self.p_global[self.p_global.shape[0] - 1, self.p_global.shape[1] - 1] = 0.01

                index_update = np.concatenate([index_update, [now_point_num + j]], axis=0)

            points_update = np.concatenate([points_update, new_frame_kp], axis=0)

        return points_update.astype(np.float32), index_update

    def tracking(self, next_img, bounding_box=None):

        matched_index, ransac_next_kp = optical_flow_matching(self.previous_img, next_img, self.previous_frame_kp)

        ransac_index = self.previous_index[matched_index]
        ransac_previous_kp = self.previous_frame_kp[matched_index]

        matched_kp, next_index, ransac_mask = run_ransac(ransac_previous_kp, ransac_next_kp, ransac_index)

        """compute inlier percentage as the measurement for tracking quality"""
        # if len(next_index) / len(previous_frame_kp) * 100 < 80:
        #     lost_cnt += 1
        # else:
        #     lost_cnt = 0
        # print("fraction: ", len(next_index) / len(previous_frame_kp))

        """
        ===============================
        1. predict step
        ===============================
        """
        """update camera pose with constant speed model"""
        # self.camera_pose += [self.delta_pan, self.delta_tilt, self.delta_zoom]
        self.camera.append(self.camera[-1])

        """update p_global"""
        q_k = 5 * np.diag([0.001, 0.001, 1])
        self.p_global[0:3, 0:3] = self.p_global[0:3, 0:3] + q_k

        """
        ===============================
        2. update step
        ===============================
        """

        height = next_img.shape[0]
        width = next_img.shape[1]

        self.ekf_update(matched_kp, next_index, height, width)

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

        self.previous_img = next_img
        self.previous_frame_kp, self.previous_index = self.add_new_points(next_img, height, width, bounding_box)

        # return previous_frame_kp, previous_index

    # def main_algorithm(self, first, step_length):
    #     """
    #     This is main function for SLAM system.
    #     Run this function to begin tracking and mapping
    #     :param first: the start frame index
    #     :param step_length: step length between consecutive frames
    #     """
    #
    #     # lost_cnt = 0
    #     # lost_frame_threshold = 3
    #     # matched_percentage = np.zeros([self.sequence.anno_size])
    #     # percentage_threshold = 80
    #
    #     # @ idealy 'sift' should can be set from a parameter
    #     # or we develop a system that uses 'sift' only
    #
    #     # This part adds the first frame to key_frame map
    #     im = self.sequence.get_image(first, 1)
    #     first_keyframe = KeyFrame(im, first, self.sequence.camera.camera_center,
    #                               self.sequence.camera.base_rotation, self.sequence.camera.principal_point[0],
    #                               self.sequence.camera.principal_point[1],
    #                               self.camera_pose[0], self.camera_pose[1], self.camera_pose[2])
    #
    #     keyframe_map.add_first_keyframe(first_keyframe)
    #
    #     for i in range(first + step_length, self.sequence.anno_size, step_length):
    #         print("=====The ", i, " iteration=====Total %d global rays\n" % len(self.ray_global))
    #
    #         """
    #         ===============================
    #         0. feature matching step
    #         ===============================
    #         """
    #         pre_img = self.sequence.get_image_gray(i - step_length, 1)
    #         next_img = self.sequence.get_image_gray(i, 1)
    #
    #         previous_frame_kp, previous_index, lost_cnt = \
    #             self.tracking(previous_frame_kp, previous_index, pre_img, next_img, lost_cnt, i)
    #
    #         """this part is for BA and relocalization"""
    #         # if matched_percentage[i] > percentage_threshold:
    #         #     # origin set to (10, 25)
    #         #     if keyframe_map.good_new_keyframe(self.camera_pose, 10, 25):
    #         #         # if keyframe_map.good_new_keyframe(self.camera_pose, 10, 15):
    #         #         print("this is keyframe:", i)
    #         #         new_keyframe = KeyFrame(self.sequence.get_image(i, 0),
    #         #                                 i, self.sequence.c, self.sequence.base_rotation, self.sequence.u,
    #         #                                 self.sequence.v, self.camera_pose[0], self.camera_pose[1],
    #         #                                 self.camera_pose[2])
    #         #         keyframe_map.add_keyframe_with_ba(new_keyframe, "./bundle_result/", verbose=True)
    #         #
    #         # elif lost_cnt > lost_frame_threshold:
    #         #     if len(keyframe_map.keyframe_list) > 1:
    #         #         self.camera_pose = relocalization_camera(keyframe_map, self.sequence.get_image(i, 0),
    #         #                                                  self.camera_pose)
    #         #         previous_frame_kp, previous_index = self.init_system(i)
    #         #         lost_cnt = 0


if __name__ == "__main__":
    sequence = SequenceManager("./two_point_calib_dataset/highlights/seq3_anno.mat",
                               "./seq3_blur",
                               "./soccer3_ground_truth.mat",
                               "./objects_soccer.mat")

    """this for soccer"""
    slam = PtzSlam()

    """this for basketball"""
    # slam = PtzSlam("./basketball/basketball/basketball_anno.mat",
    #                "./basketball/basketball/images/",
    #                "./basketball_ground_truth.mat",
    #                "./objects_basketball.mat")

    # slam = PtzSlam("/Users/jimmy/Desktop/ptz_slam_dataset/basketball/basketball_anno.mat",
    #                "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images/",
    #                "./objects_basketball.mat")

    """this for synthesized basketball court"""
    # slam = PtzSlam("./basketball/basketball/basketball_anno.mat",
    #                "./objects_basketball.mat",
    #                "./basketball/basketball/synthesize_images/")
    first_img = sequence.get_image_gray(index=0, dataset_type=1)
    first_camera = sequence.get_camera(0)
    first_bounding_box = sequence.get_bounding_box_mask(0)

    slam.init_system(first_img, first_camera, first_bounding_box)

    for i in range(1, sequence.length):
        img = sequence.get_image_gray(index=i, dataset_type=1)
        bounding_box = sequence.get_bounding_box_mask(i)
        slam.tracking(img, bounding_box)

        print("=====The ", i, " iteration=====" )

        print("%f" % (slam.camera[i].pan - sequence.ground_truth_pan[i]))
        print("%f" % (slam.camera[i].tilt - sequence.ground_truth_tilt[i]))
        print("%f" % (slam.camera[i].focal_length - sequence.ground_truth_f[i]))

    # slam.main_algorithm(first=0, step_length=1)
    #
    # draw_camera_plot(slam.sequence.ground_truth_pan, slam.sequence.ground_truth_tilt, slam.sequence.ground_truth_f,
    #                  slam.predict_pan, slam.predict_tilt, slam.predict_f)
