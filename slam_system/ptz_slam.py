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

        # global rays and covariance matrix
        self.rays = np.ndarray([0, 2])
        self.state_cov = np.zeros([3, 3])

        # the information for previous frame: image matrix, keypoints and keypoints global index.
        self.previous_img = None
        self.previous_keypoints = None
        self.previous_ray_index = None  # @todo rename it to make it clear

        self.current_camera = None  # do not use cameras[-1], or use it as fewer as possible. as it is not clear

        # map
        self.keyframe_map = Map('sift')

        # a camera list for whole sequence.
        self.cameras = []

        # speed of camera, for pan, tilt and focal length
        self.velocity = np.zeros((3, 1))
        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

    def compute_H(self, pan, tilt, focal_length, rays):
        """
        This function computes the jacobian matrix H for h(x).
        h(x) is the function from predicted state(camera pose and ray landmarks) to predicted observations.
        H helps to compute Kalman gain for the EKF.

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

        camera = copy.deepcopy(self.cameras[0])

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
        @todo many 'first' like first_img, first_camera is redundant
        just use, image, camera
        This function initializes tracking component.
        It is called: 1. At the first frame. 2. after relocalization
        :param index: begin frame index
        :return: [N, 2] array keypoints, [N] array index in global ray
        """

        # step 1: detect keypoints from image
        first_img_kp = detect_sift(first_img, 500)
        # first_img_kp = detect_orb(first_img, 300)

        # remove keypoints on players if bounding box mask is provided
        # @bug global name 'reserved_keypoints_index' is not defined
        if first_bounding_box is not None:
            first_img_kp = first_img_kp[
                reserved_keypoints_index(first_img_kp, first_bounding_box)]

        # step 2: back-project keypoint locations to rays by a known camera pose
        # use key points in first frame to get init rays
        init_rays = first_camera.back_project_to_rays(first_img_kp)

        # initialize rays
        self.rays = np.ndarray([0, 2])
        self.rays = np.row_stack([self.rays, init_rays])

        # step 3: initialize convariance matrix of states
        # some parameters are manually selected
        # @todo, note 0.001 and 1 are two parameters
        self.state_cov = 0.001 * np.eye(3 + 2 * len(self.rays))
        self.state_cov[2][2] = 1  # covariance for focal length

        # the previous frame information
        self.previous_img = first_img
        self.previous_keypoints = first_img_kp
        self.previous_index = np.array([i for i in range(len(self.rays))])

        # append the first camera to camera list
        self.cameras.append(first_camera)

    def ekf_update(self, observed_keypoints, observed_keypoint_index, height, width):
        """
        This function update global rays and covariance matrix.
        @This function is important. Please add Math and add note for variables
        @ for example: y_k, dimension, y_k is xxxx in the equation xxx
        :param observed_keypoints: matched keypoint in that frame
        :param observed_keypoint_index: matched keypoint index in global ray
        :param height: image height
        :param width: image width
        """

        # step 1: get 2d points and indexes in all landmarks with predicted camera pose
        predicted_camera = self.current_camera
        predict_keypoints, predict_keypoint_index = predicted_camera.project_rays(
            self.rays, height, width)

        # step 2: an intersection of observed keypoints and predicted keypoints
        # compute y_k: residual
        overlap1, overlap2 = get_overlap_index(observed_keypoint_index, predict_keypoint_index)
        y_k = observed_keypoints[overlap1] - predict_keypoints[overlap2]
        y_k = y_k.flatten()  # to one dimension

        # index of inlier (frame-to-frame marching) rays that from previous frame to current frame
        matched_ray_index = observed_keypoint_index[overlap1]

        # p_index is the index of rows(or cols) in p which need to be update (part of p matrix!)
        # for example, p_index = [0,1,2(pose), 3,4(ray 1), 7,8(ray 3)] means get the first and third ray.
        # step 3: extract camera pose index, and ray index in the covariance matrix
        num_ray = len(matched_ray_index)
        pose_index = np.array([0, 1, 2])
        ray_index = np.zeros(num_ray * 2)
        for j in range(num_ray):
            ray_index[2 * j + 0], ray_index[2 * j + 1] = 2 * matched_ray_index[j] + 3 + 0, 2 * matched_ray_index[
                j] + 3 + 1
        pose_ray_index = np.concatenate((pose_index, ray_index), axis=0)
        pose_ray_index = pose_ray_index.astype(np.int32)
        predicted_cov = self.state_cov[pose_ray_index][:, pose_ray_index]  # @todo, :, operator, P_k_{k-1}
        assert predicted_cov.shape[0] == pose_ray_index.shape[0] and predicted_cov.shape[1] == pose_ray_index.shape[0]

        # compute jacobi
        updated_ray = self.rays[matched_ray_index.astype(int)]
        jacobi = self.compute_H(pan=predicted_camera.pan,
                                tilt=predicted_camera.tilt,
                                focal_length=predicted_camera.focal_length,
                                rays=updated_ray)
        # get Kalman gain
        r_k = 2 * np.eye(2 * num_ray)  # todo 2 is a constant value
        s_k = np.dot(np.dot(jacobi, predicted_cov), jacobi.T) + r_k

        k_k = np.dot(np.dot(predicted_cov, jacobi.T), np.linalg.inv(s_k))

        # updated state estimate. The difference between the predicted states and the final states
        k_mul_y = np.dot(k_k, y_k)

        # update camera pose
        cur_camera = predicted_camera
        cur_camera.pan += k_mul_y[0]
        cur_camera.tilt += k_mul_y[1]
        cur_camera.focal_length += k_mul_y[2]

        self.current_camera = cur_camera  # redundant code as it is a reference

        # update speed model
        self.delta_pan, self.delta_tilt, self.delta_zoom = k_mul_y[0:3]

        # update global rays: overwrite updated ray to ray_global
        for j in range(num_ray):
            self.rays[int(matched_ray_index[j])][0:2] += k_mul_y[2 * j + 3: 2 * j + 3 + 2]

        # update global p: overwrite updated p to the p_global
        update_p = np.dot(np.eye(3 + 2 * num_ray) - np.dot(k_k, jacobi), predicted_cov)
        self.state_cov[0:3, 0:3] = update_p[0:3, 0:3]
        for j in range(num_ray):
            row1 = 3 + 2 * int(matched_ray_index[j])
            row2 = row1 + 1
            for k in range(num_ray):
                col1 = 3 + 2 * int(matched_ray_index[k])
                col2 = col1 + 1
                self.state_cov[row1, col1] = update_p[3 + 2 * j, 3 + 2 * k]
                self.state_cov[row2, col2] = update_p[3 + 2 * j + 1, 3 + 2 * k + 1]

    def remove_rays(self, ransac_mask):
        """
        remove_rays
        delete ransac outliers from global ray
        The ray is initialized by keypoint detection in the first frame.
        In the next frame, some of the keypoints are corrected matched as inliers,
        others are outliers. The outlier is associated with a ray, that ray will be removed
        Note the ray is different from the ray in the Map().

        :param ransac_mask: 0 for ourliers, 1 for inliers
        """

        # delete ray_global
        delete_index = np.ndarray([0])
        for j in range(len(ransac_mask)):
            if ransac_mask[j] == 0:
                delete_index = np.append(delete_index, j)

        self.rays = np.delete(self.rays, delete_index, axis=0)

        # delete p_global
        p_delete_index = np.ndarray([0])
        for i in range(len(delete_index)):
            p_delete_index = np.append(p_delete_index, np.array([2 * delete_index[i] + 3,
                                                                 2 * delete_index[i] + 4]))

        self.state_cov = np.delete(self.state_cov, p_delete_index, axis=0)
        self.state_cov = np.delete(self.state_cov, p_delete_index, axis=1)

    def add_rays(self, img_new, bounding_box):
        """
        Detect new keypoints in the current frame and add associated rays.
        In each frame, a number of keypoints are detected. These keypoints will
        be associated with new rays (given the camera pose). These new rays are
        added to the global ray to maintain the number of visible rays in the image.
        Otherwise, the number of rays will drop.
        :param img_new: current image

        :param bounding_box:
        :return:
        """

        # get height width of image
        height, width = img_new.shape[0:2]

        # project global_ray to image. Get existing keypoints
        keypoints, keypoints_index = self.current_camera.project_rays(
            self.rays, height, width)

        # mask to remove keypoints near existing keypoints.
        mask = np.ones(img_new.shape[0:2], np.uint8)

        for j in range(len(keypoints)):
            x, y = keypoints[j]
            up_bound = int(max(0, y - 50))
            low_bound = int(min(height, y + 50))
            left_bound = int(max(0, x - 50))
            right_bound = int(min(width, x + 50))
            mask[up_bound:low_bound, left_bound:right_bound] = 0

        new_keypoints = detect_sift(img_new, 500)
        # new_keypoints = detect_orb(img_new, 300)

        # remove keypoints in player bounding boxes
        if bounding_box is not None:
            new_keypoints = new_keypoints[reserved_keypoints_index(new_keypoints, bounding_box)]

        # remove keypoints near existing keypoints
        new_keypoints = new_keypoints[reserved_keypoints_index(new_keypoints, mask)]

        """if existing new points"""
        if new_keypoints is not None:
            new_rays = self.current_camera.back_project_to_rays(new_keypoints)

            # add new ray to ray_global, and add new rows and cols to p_global
            for j in range(len(new_rays)):
                self.rays = np.row_stack([self.rays, new_rays[j]])
                self.state_cov = np.row_stack([self.state_cov, np.zeros([2, self.state_cov.shape[1]])])
                self.state_cov = np.column_stack([self.state_cov, np.zeros([self.state_cov.shape[0], 2])])
                self.state_cov[self.state_cov.shape[0] - 2, self.state_cov.shape[1] - 2] = 0.01
                self.state_cov[self.state_cov.shape[0] - 1, self.state_cov.shape[1] - 1] = 0.01
                keypoints_index = np.append(keypoints_index, len(self.rays) - 1)

            keypoints = np.concatenate([keypoints, new_keypoints], axis=0)

        return keypoints, keypoints_index

    def tracking(self, next_img, bounding_box=None):

        matched_index, ransac_next_kp = optical_flow_matching(self.previous_img, next_img, self.previous_keypoints)

        ransac_index = self.previous_index[matched_index]
        ransac_previous_kp = self.previous_keypoints[matched_index]

        matched_kp, next_index, ransac_mask = run_ransac(ransac_previous_kp, ransac_next_kp, ransac_index)

        """compute inlier percentage as the measurement for tracking quality"""
        # if len(next_index) / len(previous_keypoints) * 100 < 80:
        #     lost_cnt += 1
        # else:
        #     lost_cnt = 0
        # print("fraction: ", len(next_index) / len(previous_keypoints))

        """
        ===============================
        1. predict step
        ===============================
        """
        """update camera pose with constant speed model"""
        self.current_camera = self.cameras[-1]
        self.cameras.append(self.current_camera)

        """update p_global"""
        q_k = 5 * np.diag([0.001, 0.001, 1])
        self.state_cov[0:3, 0:3] = self.state_cov[0:3, 0:3] + q_k

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
        # @todo bug? as ransac_mask is a local index but rays are global?
        self.remove_rays(ransac_mask)

        """
        ===============================
        4.  add new features & update previous frame
        ===============================
        """

        self.previous_img = next_img
        self.previous_keypoints, self.previous_index = self.add_rays(next_img, bounding_box)

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
    #         print("=====The ", i, " iteration=====Total %d global rays\n" % len(self.rays))
    #
    #         """
    #         ===============================
    #         0. feature matching step
    #         ===============================
    #         """
    #         pre_img = self.sequence.get_image_gray(i - step_length, 1)
    #         next_img = self.sequence.get_image_gray(i, 1)
    #
    #         previous_keypoints, previous_index, lost_cnt = \
    #             self.tracking(previous_keypoints, previous_index, pre_img, next_img, lost_cnt, i)
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
    #         #         previous_keypoints, previous_index = self.init_system(i)
    #         #         lost_cnt = 0


if __name__ == "__main__":
    """this is for soccer"""

    sequence = SequenceManager("../../dataset/soccer/seq3_anno.mat",
                               "../../dataset/soccer/images",
                               "../../dataset/soccer/soccer3_ground_truth.mat",
                               "../../dataset/soccer/objects_soccer.mat")

    """this for basketball"""
    # sequence = SequenceManager("../../dataset/basketball/basketball_anno.mat",
    #                            "../../dataset/basketball/images",
    #                            "../../dataset/basketball/basketball_ground_truth.mat",
    #                            "../../dataset/basketball/objects_basketball.mat")

    slam = PtzSlam()

    first_img = sequence.get_image_gray(index=0, dataset_type=1)
    first_camera = sequence.get_camera(0)
    first_bounding_box = sequence.get_bounding_box_mask(0)

    slam.init_system(first_img, first_camera, first_bounding_box)

    for i in range(1, sequence.length):
        img = sequence.get_image_gray(index=i, dataset_type=1)
        bounding_box = sequence.get_bounding_box_mask(i)
        slam.tracking(img, bounding_box)

        print("=====The ", i, " iteration=====")

        print("%f" % (slam.cameras[i].pan - sequence.ground_truth_pan[i]))
        print("%f" % (slam.cameras[i].tilt - sequence.ground_truth_tilt[i]))
        print("%f" % (slam.cameras[i].focal_length - sequence.ground_truth_f[i]))
