"""
Main part of our system. Ray landmarks based PTZ SLAM.

Created by Luke, 2018.9
"""

import scipy.io as sio
import cv2 as cv
import copy

from sequence_manager import SequenceManager
from scene_map import Map, RandomForestMap
from nearest_neighbor import NNBasedMap
from key_frame import KeyFrame
from relocalization import relocalization_camera
from ptz_camera import PTZCamera
from image_process import *
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
        self.previous_keypoints_index = None

        # descriptor for rays
        self.des = np.ndarray([0, 128])

        # camera object for current frame
        self.current_camera = None

        # map
        self.keyframe_map = Map('sift')

        self.rf_map = RandomForestMap()

        # a camera list for whole sequence.
        self.cameras = []

        # speed of camera, for pan, tilt and focal length
        self.velocity = np.zeros(3)

        # state: whether current frame is new keyframe.
        self.new_keyframe = False

        # state: whether current frame is lost.
        self.tracking_lost = False

        # count for bad tracking frame number. If larger than a threshold, we say this frame is lost.
        self.bad_tracking_cnt = 0

        # hyper params soccer:300 basketball: 500
        self.keypoint_num = 500

        # previous set to 2
        self.observe_var = 0.1

        self.angle_var = 0.001
        self.f_var = 1

    def compute_h_jacobian(self, pan, tilt, focal_length, rays):
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

    def init_system(self, img, camera, bounding_box=None):
        """
        This function initializes tracking component.
        It is called: 1. At the first frame. 2. after relocalization
        :param img: image to initialize system.
        :param camera:  first camera pose to initialize system.
        :param bounding_box: first bounding box matrix (optional).
        """

        # step 1: detect keypoints from image
        # first_img_kp = detect_sift(img, self.keypoint_num)
        first_img_kp, first_des = detect_compute_sift_array(img, self.keypoint_num)
        # first_img_kp = detect_orb(img, 300)
        # first_img_kp = add_gauss(first_img_kp, 50, 1280, 720)

        # x1 = copy.copy(img)
        # x2 = copy.copy(img)
        #
        # visualize_points(img, first_img_kp, (255,0,0), 5)
        # cv.imshow("test", img)
        # cv.waitKey(0)

        # # for baseline2
        # delete_index = []
        # for i in range(first_img_kp.shape[0]):
        #     world_x, world_y, _ = camera.back_project_to_3d_point(first_img_kp[i, 0], first_img_kp[i, 1])
        #     # if world_x < 0 or world_x > 118 or world_y < 0 or world_y > 70:
        #     if world_x < 0 or world_x > 25 or world_y < 0 or world_y > 18:
        #         delete_index.append(i)
        # first_img_kp = np.delete(first_img_kp, delete_index, axis=0)
        # first_des = np.delete(first_des, delete_index, axis=0)

        # visualize_points(x1, first_img_kp, (255,0,0), 5)
        # cv.imshow("test2", x1)
        # cv.waitKey(0)

        # remove keypoints on players if bounding box mask is provided
        if bounding_box is not None:
            masked_index = keypoints_masking(first_img_kp, bounding_box)
            first_img_kp = first_img_kp[masked_index]
            first_des = first_des[masked_index]

        # visualize_points(x2, first_img_kp, (255,0,0), 5)
        # cv.imshow("test3", x2)
        # cv.waitKey(0)

        # step 2: back-project keypoint locations to rays by a known camera pose
        # use key points in first frame to get init rays
        init_rays = camera.back_project_to_rays(first_img_kp)

        # initialize rays
        self.rays = np.ndarray([0, 2])
        self.rays = np.row_stack([self.rays, init_rays])

        self.des = first_des

        # step 3: initialize convariance matrix of states
        # some parameters are manually selected
        # Note 0.001 and 1 are two parameters
        self.state_cov = self.angle_var * np.eye(3 + 2 * len(self.rays))
        self.state_cov[2][2] = self.f_var  # covariance for focal length

        # the previous frame information
        self.previous_img = img
        self.previous_keypoints = first_img_kp
        self.previous_keypoints_index = np.array([i for i in range(len(self.rays))])

        # append the first camera to camera list
        self.cameras.append(camera)

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
        predicted_cov = self.state_cov[pose_ray_index][:, pose_ray_index]
        assert predicted_cov.shape[0] == pose_ray_index.shape[0] and predicted_cov.shape[1] == pose_ray_index.shape[0]

        # compute jacobi
        updated_ray = self.rays[matched_ray_index.astype(int)]
        jacobi = self.compute_h_jacobian(pan=predicted_camera.pan,
                                         tilt=predicted_camera.tilt,
                                         focal_length=predicted_camera.focal_length,
                                         rays=updated_ray)
        # get Kalman gain
        r_k = self.observe_var * np.eye(2 * num_ray)  # todo 2 is a constant value
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
        self.velocity = k_mul_y[0: 3]

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
        self.rays = np.delete(self.rays, delete_index, axis=0)
        self.des = np.delete(self.des, delete_index, axis=0)

        # delete p_global
        p_delete_index = np.ndarray([0])
        for j in range(len(delete_index)):
            p_delete_index = np.append(p_delete_index, np.array([2 * delete_index[j] + 3,
                                                                 2 * delete_index[j] + 4]))

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
        keypoints, keypoints_index = self.current_camera.project_rays(
            self.rays, height, width)

        # new_keypoints = detect_sift(img, self.keypoint_num)
        new_keypoints, new_des = detect_compute_sift_array(img, self.keypoint_num)
        # new_keypoints = detect_orb(img, 300)
        # new_keypoints = add_gauss(new_keypoints, 50, 1280, 720)

        # # for baseline2
        # delete_index = []
        # for i in range(new_keypoints.shape[0]):
        #     world_x, world_y, _ = self.current_camera.back_project_to_3d_point(new_keypoints[i, 0], new_keypoints[i, 1])
        #     # if world_x < 0 or world_x > 118 or world_y < 0 or world_y > 70:
        #     if world_x < 0 or world_x > 25 or world_y < 0 or world_y > 18:
        #         delete_index.append(i)
        # new_keypoints = np.delete(new_keypoints, delete_index, axis=0)
        # new_des = np.delete(new_des, delete_index, axis=0)

        # remove keypoints in player bounding boxes
        if bounding_box is not None:
            bounding_box_mask_index = keypoints_masking(new_keypoints, bounding_box)
            new_keypoints = new_keypoints[bounding_box_mask_index]
            new_des = new_des[bounding_box_mask_index]

        # remove keypoints near existing keypoints
        mask = np.ones(img.shape[0:2], np.uint8)

        for j in range(len(keypoints)):
            x, y = keypoints[j]
            up_bound = int(max(0, y - 50))
            low_bound = int(min(height, y + 50))
            left_bound = int(max(0, x - 50))
            right_bound = int(min(width, x + 50))
            mask[up_bound:low_bound, left_bound:right_bound] = 0

        existing_keypoints_mask_index = keypoints_masking(new_keypoints, mask)
        new_keypoints = new_keypoints[existing_keypoints_mask_index]
        new_des = new_des[existing_keypoints_mask_index]

        # check if exist new keypoints after masking.
        if new_keypoints is not None:
            new_rays = self.current_camera.back_project_to_rays(new_keypoints)

            # add new ray to ray_global, and add new rows and cols to p_global
            for j in range(len(new_rays)):
                self.rays = np.row_stack([self.rays, new_rays[j]])
                self.des = np.row_stack([self.des, new_des[j]])
                self.state_cov = np.row_stack([self.state_cov, np.zeros([2, self.state_cov.shape[1]])])
                self.state_cov = np.column_stack([self.state_cov, np.zeros([self.state_cov.shape[0], 2])])
                self.state_cov[self.state_cov.shape[0] - 2, self.state_cov.shape[1] - 2] = self.angle_var
                self.state_cov[self.state_cov.shape[0] - 1, self.state_cov.shape[1] - 1] = self.angle_var
                keypoints_index = np.append(keypoints_index, len(self.rays) - 1)

            keypoints = np.concatenate([keypoints, new_keypoints], axis=0)

        return keypoints, keypoints_index

    def tracking(self, next_img, bad_tracking_percentage, bounding_box=None):
        """
        This is function for tracking using sparse optical flow matching.
        :param next_img: image for next tracking frame
        :param bounding_box: bounding box matrix (optional)
        """

        inlier_keypoints, inlier_index, outlier_index = matching_and_ransac(
            self.previous_img, next_img, self.previous_keypoints, self.previous_keypoints_index)

        # inlier_keypoints = add_gauss(inlier_keypoints, 50, 1280, 720)

        # compute inlier percentage as the measurement for tracking quality
        tracking_percentage = len(inlier_index) / len(self.previous_keypoints) * 100
        if tracking_percentage < bad_tracking_percentage:
            self.bad_tracking_cnt += 1

        if self.bad_tracking_cnt > 3:
            # if len(self.rf_map.keyframe_list) > 8:
            self.tracking_lost = True
            self.bad_tracking_cnt = 0
        """
        ===============================
        1. predict step
        ===============================
        """

        # update camera pose with constant speed model
        self.current_camera = copy.deepcopy(self.cameras[-1])
        self.current_camera.set_ptz(self.current_camera.get_ptz() + self.velocity)

        if not self.tracking_lost:
            self.cameras.append(self.current_camera)

        # update p_global
        q_k = 5 * np.diag([self.angle_var, self.angle_var, self.f_var])
        self.state_cov[0:3, 0:3] = self.state_cov[0:3, 0:3] + q_k

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

        print("tracking", tracking_percentage)

        # if tracking_percentage > bad_tracking_percentage:
            # basketball set to (10, 25), soccer maybe (10, 15)
        if self.keyframe_map.good_new_keyframe(self.current_camera.get_ptz(), 10, 15):
            self.new_keyframe = True

        # if self.rf_map.good_keyframe(self.current_camera.get_ptz(), 10, 15):
        #     self.new_keyframe = True

    def relocalize(self, img, camera, enable_rf=False, bounding_box=None):
        """
        :param img: image to relocalize
        :param camera: lost camera to relocaize
        :return: camera after relocalize
        """

        if enable_rf:
            c = camera.camera_center
            r = camera.base_rotation
            u = camera.principal_point[0]
            v = camera.principal_point[1]
            pan = camera.pan
            tilt = camera.tilt
            focal_length = camera.focal_length
            relocalize_frame = KeyFrame(img, -1, c, r, u, v, pan, tilt, focal_length)

            kp, des = detect_compute_sift_array(img, 500)

            if bounding_box is not None:
                masked_index = keypoints_masking(kp, bounding_box)
                kp = kp[masked_index]
                des = des[masked_index]

            relocalize_frame.feature_pts = kp
            relocalize_frame.feature_des = des

            ptz = self.rf_map.relocalize(relocalize_frame, [pan, tilt, focal_length])
            camera.set_ptz(ptz)

        else:
            if len(self.keyframe_map.keyframe_list) > 1:
                lost_pose = camera.pan, camera.tilt, camera.focal_length
                relocalize_pose = relocalization_camera(self.keyframe_map, img, lost_pose)
                camera.set_ptz(relocalize_pose)
            else:
                print("Warning: Not enough keyframes for relocalization.")

        self.tracking_lost = False

        return camera

    def add_keyframe(self, img, camera, frame_index, enable_rf=False):
        """
        add new key frame.
        @todo now have not changed the KeyFrame's parameter to camera object.
        @todo Many places need to be changed if this change.
        :param img: image
        :param camera: camera object for key frame
        :param frame_index: frame index in sequence
        """
        c = camera.camera_center
        r = camera.base_rotation
        u = camera.principal_point[0]
        v = camera.principal_point[1]
        pan = camera.pan
        tilt = camera.tilt
        focal_length = camera.focal_length

        new_keyframe = KeyFrame(img, frame_index, c, r, u, v, pan, tilt, focal_length)

        if enable_rf:
            # new_keyframe.feature_pts, new_keyframe.feature_des = detect_compute_sift_array(img, 1500)
            new_keyframe.feature_pts = self.previous_keypoints
            new_keyframe.feature_des = self.des[self.previous_keypoints_index.astype(np.int)]

            self.rf_map.add_keyframe(new_keyframe)
            self.new_keyframe = False

        else:
            if frame_index == 0:
                self.keyframe_map.add_first_keyframe(new_keyframe, verbose=True)
            else:
                self.keyframe_map.add_keyframe_with_ba(new_keyframe, "./bundle_result/", verbose=True)
                self.new_keyframe = False
