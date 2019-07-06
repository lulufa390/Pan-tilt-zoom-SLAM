"""
homography based EKF tracking

Created by Luke, 2018.9
"""

import scipy.io as sio
import cv2 as cv
import copy

from sequence_manager import SequenceManager
from scene_map import Map, RandomForestMap
from key_frame import KeyFrame
from relocalization import relocalization_camera
from ptz_camera import PTZCamera
from image_process import *
from util import *
from visualize import *
from ptz_camera import *


def global_to_image(pt, homography):
    p = np.array([pt[0], pt[1], 1])
    p = np.dot(homography, p)
    image_point = np.zeros(2)
    if p[2] != 0.0:
        image_point[0] = p[0] / p[2]
        image_point[1] = p[1] / p[2]
    return image_point


def image_to_global(pt, homography):
    inv_homography = np.linalg.inv(homography)
    p = np.array([pt[0], pt[1], 1])
    p = np.dot(inv_homography, p)
    global_point = np.zeros(2)
    if p[2] != 0.0:
        global_point[0] = p[0] / p[2]
        global_point[1] = p[1] / p[2]
    return global_point


def global_to_image_array(pts, homography, height=0, width=0):
    image_points = np.ndarray([0, 2], np.float32)
    index = np.ndarray([0])

    if height != 0 and width != 0:
        for i in range(len(pts)):
            tmp = global_to_image(pts[i], homography)
            if 0 < tmp[0] < width and 0 < tmp[1] < height:
                image_points = np.row_stack([image_points, np.asarray(tmp)])
                index = np.concatenate([index, [i]], axis=0)
    else:
        for i in range(len(pts)):
            tmp = global_to_image(pts[i], homography)
            image_points = np.row_stack([image_points, np.asarray(tmp)])
            index = np.array([j for j in range(len(pts))])

    return image_points, index


def image_to_global_array(pts, homography):
    inv_homography = np.linalg.inv(homography)
    N = pts.shape[0]
    global_points = np.zeros((N, 2))
    for i in range(N):
        p = np.array([pts[i][0], pts[i][1], 1])
        p = np.dot(inv_homography, p)
        if p[2] != 0.0:
            global_points[i][0] = p[0] / p[2]
            global_points[i][1] = p[1] / p[2]

    return global_points


class HomographyEKF:
    def __init__(self):

        # global rays and covariance matrix
        self.global_keypoints = np.ndarray([0, 2])
        self.state_cov = np.zeros([8, 8])

        # the information for previous frame: image matrix, keypoints and keypoints global index.
        self.previous_img = None
        self.previous_keypoints = None
        self.previous_keypoints_index = None

        # 3*3 homography from first image plane to current frame
        self.current_homography = None

        # 3*4 matrix from world coordinate to first image plane
        self.model_to_image_homography = None

        # 3*3 homography from first image plane to each frame
        self.accumulate_homography = []

        # speed of homography parameters
        self.velocity = np.zeros(8)

        # hyper params
        # self.homo_var = 0.01
        self.homo_var = 0.01

        # self.keypoints_var = 0.001
        self.keypoints_var = 0.001

        # self.keypoint_num = 100
        self.keypoint_num = 500

        # self.observe_var = 0.00001
        self.observe_var = 0.00001

    def compute_h_jacobian(self, params, keypoints):
        """
        This function computes the jacobian matrix H for h(x).
        h(x) is the function from predicted state(camera pose and ray landmarks) to predicted observations.
        H helps to compute Kalman gain for the EKF.

        """

        keypoint_num = len(keypoints)

        delta_params = 0.0001
        delta_pixel = 0.01

        jacobi_h = np.zeros([2 * keypoint_num, 8 + 2 * keypoint_num])

        homography = np.array([[params[0], params[1], params[2]],
                               [params[3], params[4], params[5]],
                               [params[6], params[7], 1]])

        """use approximate method to compute partial derivative."""
        for i in range(keypoint_num):

            for j in range(8):
                row = j // 3
                col = j % 3
                homography_sub = homography.copy()
                homography_add = homography.copy()
                homography_sub[row, col] -= delta_params
                homography_add[row, col] += delta_params
                x_delta1, y_delta1 = global_to_image(keypoints[i], homography_sub)
                x_delta2, y_delta2 = global_to_image(keypoints[i], homography_add)

                jacobi_h[2 * i][j] = (x_delta2 - x_delta1) / (2 * delta_params)
                jacobi_h[2 * i + 1][j] = (y_delta2 - y_delta1) / (2 * delta_params)

            x_delta_x1, y_delta_x1 = global_to_image([keypoints[i, 0] - delta_pixel, keypoints[i, 1]], homography)
            x_delta_x2, y_delta_x2 = global_to_image([keypoints[i, 0] + delta_pixel, keypoints[i, 1]], homography)
            x_delta_y1, y_delta_y1 = global_to_image([keypoints[i, 0], keypoints[i, 1] - delta_pixel], homography)
            x_delta_y2, y_delta_y2 = global_to_image([keypoints[i, 0], keypoints[i, 1] + delta_pixel], homography)

            for j in range(keypoint_num):
                """only j == i, the element of H is not zero.
                the partial derivative of one 2D point to a different landmark is always zero."""
                if j == i:
                    jacobi_h[2 * i][8 + 2 * j] = (x_delta_x2 - x_delta_x1) / (2 * delta_pixel)
                    jacobi_h[2 * i][8 + 2 * j + 1] = (x_delta_y2 - x_delta_y1) / (2 * delta_pixel)

                    jacobi_h[2 * i + 1][8 + 2 * j] = (y_delta_x2 - y_delta_x1) / (2 * delta_pixel)
                    jacobi_h[2 * i + 1][8 + 2 * j + 1] = (y_delta_y2 - y_delta_y1) / (2 * delta_pixel)

        return jacobi_h

    def init_system(self, img, first_homography, bounding_box=None):
        """
        This function initializes tracking component.
        It is called: 1. At the first frame. 2. after relocalization
        :param img: image to initialize system.
        :param camera:  first camera pose to initialize system.
        :param bounding_box: first bounding box matrix (optional).
        """

        self.model_to_image_homography = first_homography

        # step 1: detect keypoints from image
        first_img_kp = detect_sift(img, self.keypoint_num)
        # first_img_kp = detect_orb(img, 300)
        # first_img_kp = add_gauss(first_img_kp, 50, 1280, 720)

        # remove keypoints on players if bounding box mask is provided
        if bounding_box is not None:
            masked_index = keypoints_masking(first_img_kp, bounding_box)
            first_img_kp = first_img_kp[masked_index]

        # initialize global keypoints
        self.global_keypoints = np.ndarray([0, 2])
        self.global_keypoints = np.row_stack([self.global_keypoints, first_img_kp])

        # step 3: initialize convariance matrix of states
        # some parameters are manually selected
        self.state_cov = self.keypoints_var * np.eye(8 + 2 * len(self.global_keypoints))
        self.state_cov[0:8, 0:8] = self.homo_var * np.eye(8)

        # the previous frame information
        self.previous_img = img
        self.previous_keypoints = first_img_kp
        self.previous_keypoints_index = np.array([i for i in range(len(self.global_keypoints))])

        # append None as there is no homography for the frame itself
        self_homography = np.eye(3)
        # self.frame_to_frame_homography.append(self_homography)
        self.accumulate_homography.append(self_homography)

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
        predict_homography = self.current_homography
        predict_keypoints, predict_keypoint_index = global_to_image_array(
            self.global_keypoints, predict_homography, height, width)

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
        homography_para_index = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        keypoint_index = np.zeros(num_ray * 2)
        for j in range(num_ray):
            keypoint_index[2 * j + 0], keypoint_index[2 * j + 1] = \
                2 * matched_ray_index[j] + 8 + 0, 2 * matched_ray_index[j] + 8 + 1
        pose_ray_index = np.concatenate((homography_para_index, keypoint_index), axis=0)
        pose_ray_index = pose_ray_index.astype(np.int32)
        predicted_cov = self.state_cov[pose_ray_index][:, pose_ray_index]
        assert predicted_cov.shape[0] == pose_ray_index.shape[0] and predicted_cov.shape[1] == pose_ray_index.shape[0]

        # compute jacobi
        updated_ray = self.global_keypoints[matched_ray_index.astype(int)]

        params = [predict_homography[0, 0], predict_homography[0, 1], predict_homography[0, 2],
                  predict_homography[1, 0], predict_homography[1, 1], predict_homography[1, 2],
                  predict_homography[2, 0], predict_homography[2, 1], ]
        jacobi = self.compute_h_jacobian(params, updated_ray)

        # get Kalman gain
        r_k = self.observe_var * np.eye(2 * num_ray)
        s_k = np.dot(np.dot(jacobi, predicted_cov), jacobi.T) + r_k

        k_k = np.dot(np.dot(predicted_cov, jacobi.T), np.linalg.pinv(s_k))

        # updated state estimate. The difference between the predicted states and the final states
        k_mul_y = np.dot(k_k, y_k)

        # update camera pose
        self.current_homography = predict_homography

        for i in range(8):
            self.current_homography[i // 3, i % 3] += k_mul_y[i]

        # update speed model
        self.velocity = k_mul_y[0: 8]

        # update global rays: overwrite updated ray to ray_global
        for j in range(num_ray):
            self.global_keypoints[int(matched_ray_index[j])][0:2] += k_mul_y[2 * j + 8: 2 * j + 8 + 2]

        # update global p: overwrite updated p to the p_global
        update_p = np.dot(np.eye(8 + 2 * num_ray) - np.dot(k_k, jacobi), predicted_cov)
        self.state_cov[0:8, 0:8] = update_p[0:8, 0:8]
        for j in range(num_ray):
            row1 = 8 + 2 * int(matched_ray_index[j])
            row2 = row1 + 1
            for k in range(num_ray):
                col1 = 8 + 2 * int(matched_ray_index[k])
                col2 = col1 + 1
                self.state_cov[row1, col1] = update_p[8 + 2 * j, 8 + 2 * k]
                self.state_cov[row2, col2] = update_p[8 + 2 * j + 1, 8 + 2 * k + 1]

    def remove_rays(self, index):
        """
        remove_rays
        delete ransac outliers from global ray
        The ray is initialized by keypoint detection in the first frame.
        In the next frame, some of the keypoints are corrected matched as inliers,
        others are outliers. The outlier is associated with a ray, that ray will be removed
        Note the ray is different from the ray in the Map().

        :param index: index in rays to be removed
        """

        # delete ray_global
        delete_index = np.array(index)
        self.global_keypoints = np.delete(self.global_keypoints, delete_index, axis=0)

        # delete p_global
        p_delete_index = np.ndarray([0])
        for j in range(len(delete_index)):
            p_delete_index = np.append(p_delete_index, np.array([8 + 2 * delete_index[j],
                                                                 8 + 2 * delete_index[j] + 1]))

        self.state_cov = np.delete(self.state_cov, p_delete_index, axis=0)
        self.state_cov = np.delete(self.state_cov, p_delete_index, axis=1)

    def add_rays(self, img, bounding_box):
        """
        Detect new keypoints in the current frame and add associated rays.
        In each frame, a number of keypoints are detected. These keypoints will
        be associated with new rays (given the camera pose). These new rays are
        added to the global ray to maintain the number of visible rays in the image.
        Otherwise, the number of rays will drop.
        :param img: current image
        :param bounding_box: matrix same size as img. 0 is on players, 1 is out of players.
        :return: keypoints and corresponding global indexes
        """

        # get height width of image
        height, width = img.shape[0:2]

        # project global_ray to image. Get existing keypoints
        keypoints, keypoints_index = global_to_image_array(self.global_keypoints, self.current_homography,
                                                           height, width)

        new_keypoints = detect_sift(img, self.keypoint_num)
        # new_keypoints = detect_orb(img, 300)
        # new_keypoints = add_gauss(new_keypoints, 50, 1280, 720)

        # remove keypoints in player bounding boxes
        if bounding_box is not None:
            bounding_box_mask_index = keypoints_masking(new_keypoints, bounding_box)
            new_keypoints = new_keypoints[bounding_box_mask_index]

        # remove keypoints near existing keypoints
        mask = np.ones(img.shape[0:2], np.uint8)

        neigibor_size = 20

        for j in range(len(keypoints)):
            x, y = keypoints[j]
            up_bound = int(max(0, y - neigibor_size))
            low_bound = int(min(height, y + neigibor_size))
            left_bound = int(max(0, x - neigibor_size))
            right_bound = int(min(width, x + neigibor_size))
            mask[up_bound:low_bound, left_bound:right_bound] = 0

        existing_keypoints_mask_index = keypoints_masking(new_keypoints, mask)
        new_keypoints = new_keypoints[existing_keypoints_mask_index]

        # check if exist new keypoints after masking.
        if new_keypoints is not None:
            new_rays = image_to_global_array(new_keypoints, self.current_homography)

            # add new ray to ray_global, and add new rows and cols to p_global
            for j in range(len(new_rays)):
                self.global_keypoints = np.row_stack([self.global_keypoints, new_rays[j]])
                self.state_cov = np.row_stack([self.state_cov, np.zeros([2, self.state_cov.shape[1]])])
                self.state_cov = np.column_stack([self.state_cov, np.zeros([self.state_cov.shape[0], 2])])
                self.state_cov[self.state_cov.shape[0] - 2, self.state_cov.shape[1] - 2] = self.keypoints_var
                self.state_cov[self.state_cov.shape[0] - 1, self.state_cov.shape[1] - 1] = self.keypoints_var
                keypoints_index = np.append(keypoints_index, len(self.global_keypoints) - 1)

            keypoints = np.concatenate([keypoints, new_keypoints], axis=0)

        return keypoints, keypoints_index

    def tracking(self, next_img, bounding_box=None):
        """
        This is function for tracking using sparse optical flow matching.
        :param next_img: image for next tracking frame
        :param bounding_box: bounding box matrix (optional)
        """

        inlier_keypoints, inlier_index, outlier_index = matching_and_ransac(
            self.previous_img, next_img, self.previous_keypoints, self.previous_keypoints_index)

        # inlier_keypoints = add_gauss(inlier_keypoints, 50, 1280, 720)

        """
        ===============================
        1. predict step
        ===============================
        """

        # update camera pose with constant speed model
        self.current_homography = self.accumulate_homography[-1].copy()
        for i in range(8):
            self.current_homography[i // 3, i % 3] += self.velocity[i]

        # update p_global
        q_k = 5 * np.diag([self.homo_var for _ in range(8)])
        self.state_cov[0:8, 0:8] = self.state_cov[0:8, 0:8] + q_k

        """
        ===============================
        2. update step
        ===============================
        """

        height, width = next_img.shape[0:2]
        self.ekf_update(inlier_keypoints, inlier_index, height, width)

        """
        ===============================
        3. delete outlier_index
        ===============================
        """

        self.remove_rays(outlier_index)

        """
        ===============================
        4.  add new features & update previous frame
        ===============================
        """

        self.previous_img = next_img
        self.previous_keypoints, self.previous_keypoints_index = self.add_rays(next_img, bounding_box)

        self.accumulate_homography.append(self.current_homography)


def soccer3_test():
    sequence = SequenceManager("../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_330",
                               "../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_player_bounding_box.mat")

    line_index, points = load_model("../../dataset/soccer_dataset/highlights_soccer_model.mat")

    first_frame_ptz = (sequence.ground_truth_pan[0],
                       sequence.ground_truth_tilt[0],
                       sequence.ground_truth_f[0])

    first_camera = sequence.camera
    first_camera.set_ptz(first_frame_ptz)

    # 3*4 projection matrix for 1st frame
    first_frame_mat = first_camera.projection_matrix
    first_frame = sequence.get_image_gray(index=0, dataset_type=1)
    first_bounding_box = sequence.get_bounding_box_mask(0)
    # img = project_with_homography(first_frame_mat, points, line_index, first_frame)
    #
    # cv.imshow("image", img)
    # cv.waitKey()

    homography_ekf = HomographyEKF()

    homography_ekf.init_system(first_frame, first_frame_mat, first_bounding_box)

    # tracking_obj = HomographyTracking(first_frame, first_frame_mat)

    points3d_on_field = uniform_point_sample_on_field(118, 70, 50, 25)

    pan = [first_frame_ptz[0]]
    tilt = [first_frame_ptz[1]]
    f = [first_frame_ptz[2]]

    for i in range(1, sequence.length):
        next_frame = sequence.get_image_gray(index=i, dataset_type=1)
        next_bounding_box = sequence.get_bounding_box_mask(i)
        homography_ekf.tracking(next_frame, next_bounding_box)

        # img = project_with_homography(
        #     np.dot(homography_ekf.accumulate_homography[-1], homography_ekf.model_to_image_homography),
        #     points, line_index, next_frame)

        # compute ptz

        first_camera.set_ptz((pan[-1], tilt[-1], f[-1]))

        current_homography = np.dot(homography_ekf.accumulate_homography[-1], homography_ekf.model_to_image_homography)

        pose = estimate_camera_from_homography(current_homography, first_camera, points3d_on_field)

        print("-----" + str(i) + "--------")

        # print("hompgraphy:", homography_ekf.accumulate_homography)

        print(pose)

        # first_camera.set_ptz(pose)
        # img2 = project_with_PTZCamera(first_camera, points, line_index, next_frame)

        print("%f" % (pose[0] - sequence.ground_truth_pan[i]))
        print("%f" % (pose[1] - sequence.ground_truth_tilt[i]))
        print("%f" % (pose[2] - sequence.ground_truth_f[i]))

        pan.append(pose[0])
        tilt.append(pose[1])
        f.append(pose[2])

        # cv.imshow("image", img)
        # cv.imshow("image2", img2)
        # cv.waitKey(0)

    save_camera_pose(np.array(pan), np.array(tilt), np.array(f),
                     "./result.mat")


def basketball_test():
    sequence = SequenceManager("../../dataset/basketball/ground_truth.mat",
                               "../../dataset/basketball/images",
                               "../../dataset/basketball/ground_truth.mat",
                               "../../dataset/basketball/bounding_box.mat")

    # line_index, points = load_model("../../dataset/soccer_dataset/highlights_soccer_model.mat")

    begin_frame = 0

    first_frame_ptz = (sequence.ground_truth_pan[begin_frame],
                       sequence.ground_truth_tilt[begin_frame],
                       sequence.ground_truth_f[begin_frame])

    first_camera = sequence.camera
    first_camera.set_ptz(first_frame_ptz)

    # 3*4 projection matrix for 1st frame
    first_frame_mat = first_camera.projection_matrix
    first_frame = sequence.get_image_gray(index=begin_frame, dataset_type=0)
    first_bounding_box = sequence.get_bounding_box_mask(begin_frame)
    # img = project_with_homography(first_frame_mat, points, line_index, first_frame)
    #
    # cv.imshow("image", img)
    # cv.waitKey()

    homography_ekf = HomographyEKF()

    homography_ekf.init_system(first_frame, first_frame_mat, first_bounding_box)

    # tracking_obj = HomographyTracking(first_frame, first_frame_mat)

    points3d_on_field = uniform_point_sample_on_field(25, 18, 25, 18)

    pan = [first_frame_ptz[0]]
    tilt = [first_frame_ptz[1]]
    f = [first_frame_ptz[2]]

    for i in range(1, sequence.length):
        next_frame = sequence.get_image_gray(index=i, dataset_type=0)
        next_bounding_box = sequence.get_bounding_box_mask(i)
        homography_ekf.tracking(next_frame, next_bounding_box)

        # img = project_with_homography(
        #     np.dot(homography_ekf.accumulate_homography[-1], homography_ekf.model_to_image_homography),
        #     points, line_index, next_frame)

        # compute ptz

        first_camera.set_ptz((pan[-1], tilt[-1], f[-1]))

        current_homography = np.dot(homography_ekf.accumulate_homography[-1], homography_ekf.model_to_image_homography)

        pose = estimate_camera_from_homography(current_homography, first_camera, points3d_on_field)

        print("-----" + str(i) + "--------")

        # print("hompgraphy:", homography_ekf.accumulate_homography)

        print(pose)

        # first_camera.set_ptz(pose)
        # img2 = project_with_PTZCamera(first_camera, points, line_index, next_frame)

        print("%f" % (pose[0] - sequence.ground_truth_pan[i]))
        print("%f" % (pose[1] - sequence.ground_truth_tilt[i]))
        print("%f" % (pose[2] - sequence.ground_truth_f[i]))

        pan.append(pose[0])
        tilt.append(pose[1])
        f.append(pose[2])

        # cv.imshow("image", img)
        # cv.imshow("image2", img2)
        # cv.waitKey(0)

    save_camera_pose(np.array(pan), np.array(tilt), np.array(f), "./bs4_result.mat")


def synthesized_test():
    sequence = SequenceManager(annotation_path="../../dataset/basketball/ground_truth.mat",
                               image_path="../../dataset/synthesized/images")

    gt_pan, gt_tilt, gt_f = load_camera_pose("../../dataset/synthesized/synthesize_ground_truth.mat", separate=True)

    line_index, points = load_model("../../dataset/basketball/basketball_model.mat")

    begin_frame = 2400

    first_frame_ptz = (gt_pan[begin_frame],
                       gt_tilt[begin_frame],
                       gt_f[begin_frame])

    first_camera = sequence.camera
    first_camera.set_ptz(first_frame_ptz)

    # print(first_camera.project_ray((0, 0)))
    # print(first_camera.project_ray((10, 10)))
    # print(first_camera.project_ray((10.1, 10)))

    # 3*4 projection matrix for 1st frame
    first_frame_mat = first_camera.projection_matrix
    first_frame = sequence.get_image_gray(index=begin_frame, dataset_type=2)
    # first_bounding_box = sequence.get_bounding_box_mask(0)
    # img = project_with_homography(first_frame_mat, points, line_index, first_frame)
    #
    # cv.imshow("image", img)
    # cv.waitKey()

    # test_camera = copy.deepcopy(first_camera)
    # test_camera.set_ptz((gt_pan[5],
    #                    gt_tilt[5],
    #                    gt_f[5]))
    # re = compute_reprojection_error(first_frame, first_camera, test_camera)

    homography_ekf = HomographyEKF()

    homography_ekf.init_system(first_frame, first_frame_mat)

    # tracking_obj = HomographyTracking(first_frame, first_frame_mat)

    points3d_on_field = uniform_point_sample_on_field(25, 18, 25, 18)

    pan = [first_frame_ptz[0]]
    tilt = [first_frame_ptz[1]]
    f = [first_frame_ptz[2]]

    for i in range(2401, 3000, 1):
        next_frame = sequence.get_image_gray(index=i, dataset_type=2)

        homography_ekf.tracking(next_frame)

        # img = project_with_homography(
        #     np.dot(homography_ekf.accumulate_homography[-1], homography_ekf.model_to_image_homography),
        #     points, line_index, next_frame)

        # compute ptz

        first_camera.set_ptz((pan[-1], tilt[-1], f[-1]))

        current_homography = np.dot(homography_ekf.accumulate_homography[-1],
                                    homography_ekf.model_to_image_homography)

        pose = estimate_camera_from_homography(current_homography, first_camera, points3d_on_field)

        print("-----" + str(i) + "--------")

        print(len(homography_ekf.previous_keypoints))

        # print("homo:", homography_ekf.accumulate_homography[-1])

        print(pose)

        # first_camera.set_ptz(pose)
        # img2 = project_with_PTZCamera(first_camera, points, line_index, next_frame)

        print("%f" % (pose[0] - gt_pan[i]))
        print("%f" % (pose[1] - gt_tilt[i]))
        print("%f" % (pose[2] - gt_f[i]))

        pan.append(pose[0])
        tilt.append(pose[1])
        f.append(pose[2])

        # cv.imshow("image", img)
        # cv.imshow("image2", img2)
        # cv.waitKey(0)

    save_camera_pose(np.array(pan), np.array(tilt), np.array(f),
                     "C:/graduate_design/experiment_result/baseline2/synthesized/new/homography-2400.mat")


if __name__ == "__main__":
    # soccer3_test()
    # synthesized_test()
    basketball_test()
