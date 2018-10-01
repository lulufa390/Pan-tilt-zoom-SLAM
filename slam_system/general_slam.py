"""
This is a adaptation of standard 3D based SLAM to PTZ camera.
Ray landmarks are replaced by 3D points.

Create by Luke, 2018.9
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from transformation import TransFunction
from image_process import *
from sequence_manager import SequenceManager
from image_generator import ImageGenerator
from util import *


class GeneralSlam:
    """This is a class for standard 3D SLAM based on EKF.
    Use main_algorithm function to run tracking component.
    """

    def __init__(self, annotation_path, bounding_box_path, image_path):
        """
        :param annotation_path: the path of annotation file (ground truth).
        :param bounding_box_path: the path of player bounding box (player detection result).
        :param image_path: the image folder for video sequence.
        """

        """synthesized court does not need bounding box"""
        self.sequence = SequenceManager(annotation_path, image_path, bounding_box_path)

        self.sequence_length = self.sequence.anno_size
        # self.sequence_length = 333

        """parameters to be updated"""
        self.camera_pose = np.ndarray([3])
        self.delta_pan, self.delta_tilt, self.delta_zoom = [0, 0, 0]

        """global rays and covariance matrix"""
        self.ray_global = np.ndarray([0, 3])
        self.p_global = np.zeros([3, 3])

        """set the ground truth camera pose for whole sequence"""
        self.ground_truth_pan = np.ndarray([self.sequence.anno_size])
        self.ground_truth_tilt = np.ndarray([self.sequence.anno_size])
        self.ground_truth_f = np.ndarray([self.sequence.anno_size])
        for i in range(self.sequence.anno_size):
            self.ground_truth_pan[i], self.ground_truth_tilt[i], self.ground_truth_f[i] \
                = self.sequence.get_ptz(i)

        """camera pose sequence (basketball)"""
        self.predict_pan = np.zeros([self.sequence_length])
        self.predict_tilt = np.zeros([self.sequence_length])
        self.predict_f = np.zeros([self.sequence_length])

    def compute_new_jacobi(self, camera_pan, camera_tilt, foc, center, rotation, rays):
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

        delta_len = 0.01

        jacobi_h = np.ndarray([2 * ray_num, 3 + 3 * ray_num])

        """use approximate method to compute partial derivative."""
        for i in range(ray_num):
            x_delta_pan1, y_delta_pan1 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan - delta_angle, camera_tilt, center, rotation, rays[i])

            x_delta_pan2, y_delta_pan2 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan + delta_angle, camera_tilt, center, rotation, rays[i])

            x_delta_tilt1, y_delta_tilt1 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan, camera_tilt - delta_angle, center, rotation, rays[i])

            x_delta_tilt2, y_delta_tilt2 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan, camera_tilt + delta_angle, center, rotation, rays[i])

            x_delta_f1, y_delta_f1 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc - delta_f, camera_pan, camera_tilt, center, rotation, rays[i])

            x_delta_f2, y_delta_f2 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc + delta_f, camera_pan, camera_tilt, center, rotation, rays[i])

            x_delta_x1, y_delta_x1 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan, camera_tilt, center, rotation,
                rays[i] - np.array([delta_len, 0, 0]))

            x_delta_x2, y_delta_x2 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan, camera_tilt, center, rotation,
                rays[i] + np.array([delta_len, 0, 0]))

            x_delta_y1, y_delta_y1 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan, camera_tilt, center, rotation,
                rays[i] - np.array([0, delta_len, 0]))
            x_delta_y2, y_delta_y2 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan, camera_tilt, center, rotation,
                rays[i] + np.array([0, delta_len, 0]))

            x_delta_z1, y_delta_z1 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan, camera_tilt, center, rotation,
                rays[i] - np.array([0, 0, delta_len]))
            x_delta_z2, y_delta_z2 = TransFunction.from_3d_to_2d(
                self.sequence.u, self.sequence.v, foc, camera_pan, camera_tilt, center, rotation,
                rays[i] + np.array([0, 0, delta_len]))

            jacobi_h[2 * i][0] = (x_delta_pan2 - x_delta_pan1) / (2 * delta_angle)
            jacobi_h[2 * i][1] = (x_delta_tilt2 - x_delta_tilt1) / (2 * delta_angle)
            jacobi_h[2 * i][2] = (x_delta_f2 - x_delta_f1) / (2 * delta_f)

            jacobi_h[2 * i + 1][0] = (y_delta_pan2 - y_delta_pan1) / (2 * delta_angle)
            jacobi_h[2 * i + 1][1] = (y_delta_tilt2 - y_delta_tilt1) / (2 * delta_angle)
            jacobi_h[2 * i + 1][2] = (y_delta_f2 - y_delta_f1) / (2 * delta_f)

            for j in range(ray_num):
                if j == i:
                    jacobi_h[2 * i][3 + 3 * j] = (x_delta_x2 - x_delta_x1) / (2 * delta_len)
                    jacobi_h[2 * i][3 + 3 * j + 1] = (x_delta_y2 - x_delta_y1) / (2 * delta_len)
                    jacobi_h[2 * i][3 + 3 * j + 2] = (x_delta_z2 - x_delta_z1) / (2 * delta_len)

                    jacobi_h[2 * i + 1][3 + 3 * j] = (y_delta_x2 - y_delta_x1) / (2 * delta_len)
                    jacobi_h[2 * i + 1][3 + 3 * j + 1] = (y_delta_y2 - y_delta_y1) / (2 * delta_len)
                    jacobi_h[2 * i + 1][3 + 3 * j + 2] = (y_delta_z2 - y_delta_z1) / (2 * delta_len)

                else:
                    jacobi_h[2 * i][3 + 3 * j] = jacobi_h[2 * i][3 + 3 * j + 1] = jacobi_h[2 * i][3 + 3 * j + 2] = \
                        jacobi_h[2 * i + 1][3 + 3 * j] = jacobi_h[2 * i + 1][3 + 3 * j + 1] = jacobi_h[2 * i + 1][
                        3 + 3 * j + 2] = 0

        return jacobi_h

    def init_system(self, index):
        """
        Use this function to begin tracking or recover tracking after relocalization.
        :param index: the begin frame index for system.
        """

        """first frame to initialize global_rays"""
        begin_frame = self.sequence.get_image_gray(index)

        begin_frame_kp = detect_sift(begin_frame)

        """remove keypoints on player"""
        begin_frame_kp = begin_frame_kp[
            remove_player_feature(begin_frame_kp, self.sequence.get_bounding_box_mask(index))]

        """remove keypoints out of ground"""
        camera = np.array([self.sequence.u, self.sequence.v, self.sequence.get_ptz(index)[2],
                           self.sequence.get_ptz(index)[0], self.sequence.get_ptz(index)[1],
                           self.sequence.c[0], self.sequence.c[1], self.sequence.c[2]])
        ground_map = np.ones(shape=self.sequence.get_image_gray(index).shape)
        court_mask = ImageGenerator().generate_image(camera, ground_map)
        begin_frame_kp = begin_frame_kp[
            remove_player_feature(begin_frame_kp, court_mask)]

        """use key points in first frame to get init rays"""
        init_rays = TransFunction.get_3ds_from_observation(self.camera_pose[0], self.camera_pose[1],
                                                           self.camera_pose[2],
                                                           begin_frame_kp, self.sequence.u, self.sequence.v,
                                                           self.sequence.c, self.sequence.base_rotation)

        """initialize ray_global"""
        self.ray_global = np.ndarray([0, 3])
        self.ray_global = np.row_stack([self.ray_global, init_rays])

        """initialize global p using global rays"""
        self.p_global = np.zeros([3, 3])
        self.p_global = 0.001 * np.eye(3 + 3 * len(self.ray_global))
        self.p_global[2][2] = 1

        """set keypoints in previous frame for next frame"""
        previous_frame_kp = begin_frame_kp
        previous_index = np.array([i for i in range(len(self.ray_global))])

        """save estimate pose"""
        self.predict_pan[index], self.predict_tilt[index], self.predict_f[index] = self.camera_pose

        return previous_frame_kp, previous_index

    def ekf_update(self, i, matched_kp, next_index):
        """
        :param i: index for frame
        :param matched_kp: matched keypoint in that frame
        :param next_index: matched keypoint index in global 3d_point
        """
        # get 2d points, rays and indexes in all landmarks with predicted camera pose
        predict_points, inner_point_index = TransFunction.get_observation_from_3ds(
            self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], self.ray_global,
            self.sequence.u, self.sequence.v, self.sequence.c, self.sequence.base_rotation,
            self.sequence.height, self.sequence.width)

        # compute y_k
        overlap1, overlap2 = get_overlap_index(next_index, inner_point_index)

        y_k = matched_kp[overlap1] - predict_points[overlap2]
        y_k = y_k.flatten()

        matched_inner_point_index = next_index[overlap1]

        # get p matrix for this iteration from p_global
        p_index = np.array([0, 1, 2])
        for j in range(len(matched_inner_point_index)):
            p_index = np.append(p_index, np.array([3 * matched_inner_point_index[j] + 3,
                                                   3 * matched_inner_point_index[j] + 4,
                                                   3 * matched_inner_point_index[j] + 5]))
        p_index = p_index.astype(np.int32)

        p = self.p_global[p_index][:, p_index]

        # compute jacobi
        jacobi = self.compute_new_jacobi(camera_pan=self.camera_pose[0], camera_tilt=self.camera_pose[1],
                                         foc=self.camera_pose[2], center=self.sequence.c,
                                         rotation=self.sequence.base_rotation,
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
            self.ray_global[int(matched_inner_point_index[j])][0:3] += k_mul_y[3 * j + 3: 3 * j + 6]

        # update global p
        update_p = np.dot(np.eye(3 + 3 * len(matched_inner_point_index)) - np.dot(k_k, jacobi), p)
        self.p_global[0:3, 0:3] = update_p[0:3, 0:3]
        for j in range(len(matched_inner_point_index)):
            for k in range(len(matched_inner_point_index)):
                self.p_global[
                    3 + 2 * int(matched_inner_point_index[j]), 3 + 2 * int(matched_inner_point_index[k])] = \
                    update_p[3 + 2 * j, 3 + 2 * k]
                self.p_global[
                    3 + 2 * int(matched_inner_point_index[j]) + 1, 3 + 2 * int(matched_inner_point_index[k]) + 1] = \
                    update_p[3 + 2 * j + 1, 3 + 2 * k + 1]

                self.p_global[
                    3 + 2 * int(matched_inner_point_index[j]) + 2, 3 + 2 * int(matched_inner_point_index[k]) + 2] = \
                    update_p[3 + 2 * j + 2, 3 + 2 * k + 2]

    def delete_outliers(self, ransac_mask):
        """
        delete ransac outliers from global ray
        :param ransac_mask: 0 for ourliers, 1 for inliers
        """

        # delete global_ray
        delete_index = np.ndarray([0])
        for j in range(len(ransac_mask)):
            if ransac_mask[j] == 0:
                delete_index = np.append(delete_index, j)

        self.ray_global = np.delete(self.ray_global, delete_index, axis=0)

        # delete p_global
        p_delete_index = np.ndarray([0])
        for i in range(len(delete_index)):
            p_delete_index = np.append(p_delete_index, np.array([3 * delete_index[i] + 3,
                                                                 3 * delete_index[i] + 4,
                                                                 3 * delete_index[i] + 5]))

        self.p_global = np.delete(self.p_global, p_delete_index, axis=0)
        self.p_global = np.delete(self.p_global, p_delete_index, axis=1)

    def add_new_points(self, i):
        """
        :param i: frame index.
        :return: previous frame keypoints and index.
        """
        points_update, index_update = TransFunction.get_observation_from_3ds(
            self.camera_pose[0], self.camera_pose[1], self.camera_pose[2], self.ray_global,
            self.sequence.u, self.sequence.v, self.sequence.c, self.sequence.base_rotation,
            self.sequence.height, self.sequence.width)

        img_new = self.sequence.get_image_gray(i)

        """set the mask"""
        mask = np.ones(img_new.shape, np.uint8)
        for j in range(len(points_update)):
            x, y = points_update[j]
            up_bound = int(max(0, y - 50))
            low_bound = int(min(self.sequence.height, y + 50))
            left_bound = int(max(0, x - 50))
            right_bound = int(min(self.sequence.width, x + 50))
            mask[up_bound:low_bound, left_bound:right_bound] = 0

        all_new_frame_kp = detect_sift(img_new)

        all_new_frame_kp = all_new_frame_kp[
            remove_player_feature(all_new_frame_kp, self.sequence.get_bounding_box_mask(i))]

        """begin only feature on ground!"""
        camera = np.array([self.sequence.u, self.sequence.v, self.sequence.get_ptz(i)[2],
                           self.sequence.get_ptz(i)[0], self.sequence.get_ptz(i)[1],
                           self.sequence.c[0], self.sequence.c[1], self.sequence.c[2]])
        ground_map = np.ones(shape=self.sequence.get_image_gray(i).shape)
        court_mask = ImageGenerator().generate_image(camera, ground_map)
        all_new_frame_kp = all_new_frame_kp[
            remove_player_feature(all_new_frame_kp, court_mask)]
        """end only feature on ground!"""

        new_frame_kp = np.ndarray([0, 2])
        """use mask to remove feature points near existing points"""
        for j in range(len(all_new_frame_kp)):
            if mask[int(all_new_frame_kp[j, 1]), int(all_new_frame_kp[j, 0])] == 1:
                new_frame_kp = np.concatenate([new_frame_kp, (all_new_frame_kp[j]).reshape([1, 2])], axis=0)

        """if existing new points"""
        if new_frame_kp is not None:

            new_rays = TransFunction.get_3ds_from_observation(self.camera_pose[0], self.camera_pose[1],
                                                              self.camera_pose[2],
                                                              new_frame_kp, self.sequence.u, self.sequence.v,
                                                              self.sequence.c, self.sequence.base_rotation)

            now_point_num = len(self.ray_global)

            """add to global ray and covariance matrix"""
            for j in range(len(new_rays)):
                self.ray_global = np.row_stack([self.ray_global, new_rays[j]])
                self.p_global = np.row_stack([self.p_global, np.zeros([3, self.p_global.shape[1]])])
                self.p_global = np.column_stack([self.p_global, np.zeros([self.p_global.shape[0], 3])])

                self.p_global[self.p_global.shape[0] - 3, self.p_global.shape[1] - 3] = 0.01
                self.p_global[self.p_global.shape[0] - 2, self.p_global.shape[1] - 2] = 0.01
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
        self.camera_pose = self.sequence.get_ptz(first)
        previous_frame_kp, previous_index = self.init_system(first)

        for i in range(first + step_length, self.sequence_length, step_length):
            print("=====The ", i, " iteration=====Total %d global rays\n" % len(self.ray_global))

            """
            ===============================
            0. feature matching step
            ===============================
            """
            pre_img = self.sequence.get_image_gray(i - step_length)
            next_img = self.sequence.get_image_gray(i)

            matched_index, ransac_next_kp = optical_flow_matching(pre_img, next_img, previous_frame_kp)

            ransac_index = previous_index[matched_index]
            ransac_previous_kp = previous_frame_kp[matched_index]

            matched_kp, next_index, ransac_mask = run_ransac(ransac_previous_kp, ransac_next_kp, ransac_index)

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

    def output_camera_error(self, now_index):
        """
        output the error of camera pose compared to ground truth
        :param now_index: frame index
        """
        ground_truth = np.array(
            [self.ground_truth_pan[now_index], self.ground_truth_tilt[now_index], self.ground_truth_f[now_index]])
        pan, tilt, f = self.camera_pose - ground_truth
        print("%.3f %.3f, %.1f" % (pan, tilt, f), "\n")

    def draw_camera_plot(self):
        """
        draw plot for ground truth and estimated camera pose.
        """
        plt.figure("pan percentage error")
        x = np.array([i for i in range(self.sequence_length)])
        plt.plot(x, (self.predict_pan - self.ground_truth_pan) / self.ground_truth_pan * 100, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("error %")
        plt.legend(loc="best")

        plt.figure("tilt percentage error")
        x = np.array([i for i in range(self.sequence_length)])
        plt.plot(x, (self.predict_tilt - self.ground_truth_tilt) / self.ground_truth_tilt * 100, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("error %")
        plt.legend(loc="best")

        plt.figure("f percentage error")
        x = np.array([i for i in range(self.sequence_length)])
        # plt.plot(x, self.ground_truth_f, 'r', label='ground truth')
        plt.plot(x, (self.predict_f - self.ground_truth_f) / self.ground_truth_f * 100, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("error %")
        plt.legend(loc="best")

        """absolute value"""
        plt.figure("pan")
        x = np.array([i for i in range(self.sequence_length)])
        plt.plot(x, self.ground_truth_pan, 'r', label='ground truth')
        plt.plot(x, self.predict_pan, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("pan angle")
        plt.legend(loc="best")

        plt.figure("tilt")
        x = np.array([i for i in range(self.sequence_length)])
        plt.plot(x, self.ground_truth_tilt, 'r', label='ground truth')
        plt.plot(x, self.predict_tilt, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("tilt angle")
        plt.legend(loc="best")

        plt.figure("f")
        x = np.array([i for i in range(self.sequence_length)])
        plt.plot(x, self.ground_truth_f, 'r', label='ground truth')
        plt.plot(x, self.predict_f, 'b', label='predict')
        plt.xlabel("frame")
        plt.ylabel("f")
        plt.legend(loc="best")

        """this part is for soccer specifically
        because soccer annotations are less than images"""
        # plt.figure("pan")
        # x1 = np.array([6 * i for i in range(self.sequence_length // 6)])
        # x2 = np.array([i for i in range(self.sequence_length)])
        # plt.plot(x1, self.ground_truth_pan[:self.sequence_length // 6], 'r', label='ground truth')
        # plt.plot(x2, self.predict_pan, 'b', label='predict')
        # plt.xlabel("frame")
        # plt.ylabel("pan angle")
        # plt.legend(loc="best")
        #
        # plt.figure("tilt")
        # x1 = np.array([6 * i for i in range(self.sequence_length // 6)])
        # x2 = np.array([i for i in range(self.sequence_length)])
        # plt.plot(x1, self.ground_truth_tilt[:self.sequence_length // 6], 'r', label='ground truth')
        # plt.plot(x2, self.predict_tilt, 'b', label='predict')
        # plt.xlabel("frame")
        # plt.ylabel("tilt angle")
        # plt.legend(loc="best")
        #
        # plt.figure("f")
        # x1 = np.array([6 * i for i in range(self.sequence_length // 6)])
        # x2 = np.array([i for i in range(self.sequence_length)])
        # plt.plot(x1, self.ground_truth_f[:self.sequence_length // 6], 'r', label='ground truth')
        # plt.plot(x2, self.predict_f, 'b', label='predict')
        # plt.xlabel("frame")
        # plt.ylabel("f")
        # plt.legend(loc="best")

        plt.show()

    def save_camera_to_mat(self):
        """
        save ground truth and estimated camera pose into .mat file.
        :return:
        """
        camera_pose = dict()

        camera_pose['ground_truth_pan'] = self.ground_truth_pan
        camera_pose['ground_truth_tilt'] = self.ground_truth_tilt
        camera_pose['ground_truth_f'] = self.ground_truth_f

        camera_pose['predict_pan'] = self.predict_pan
        camera_pose['predict_tilt'] = self.predict_tilt
        camera_pose['predict_f'] = self.predict_f

        sio.savemat('camera_pose.mat', mdict=camera_pose)

    def load_camera_mat(self, path):
        """
        load ground truth and estimated camera pose
        :param path: .mat file path
        """
        camera_pos = sio.loadmat(path)
        self.predict_pan = camera_pos['predict_pan'].squeeze()
        self.predict_tilt = camera_pos['predict_tilt'].squeeze()
        self.predict_f = camera_pos['predict_f'].squeeze()

        self.ground_truth_pan = camera_pos['ground_truth_pan'].squeeze()
        self.ground_truth_tilt = camera_pos['ground_truth_tilt'].squeeze()
        self.ground_truth_f = camera_pos['ground_truth_f'].squeeze()


if __name__ == "__main__":
    """this for soccer"""
    # slam = GeneralSlam("./two_point_calib_dataset/highlights/seq3_anno.mat",
    #                    "./objects_soccer.mat",
    #                    "./seq3_blur/")

    """this for basketball"""
    slam = GeneralSlam("./basketball/basketball/basketball_anno.mat",
                       "./objects_basketball.mat",
                       "./basketball/basketball/images/")

    slam.main_algorithm(first=0, step_length=1)

    slam.draw_camera_plot()
    slam.save_camera_to_mat()
