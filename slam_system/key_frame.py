"""
Keyframe class.

Created by Luke, 2018.9
"""

import scipy.io as sio
import numpy as np
import cv2 as cv
from image_process import detect_compute_sift_array, visualize_points
from util import *

class KeyFrame:
    """This is a class for keyframe in mapping."""

    def __init__(self, img, img_index, center, rotation, u, v, pan, tilt, f):
        """
        :param img: image array for keyframe
        :param img_index: index in sequence
        :param center: camera center array [3]
        :param rotation: base rotation matrix array [3, 3]
        :param u: parameter u
        :param v: parameter v
        :param pan: camera pose, pan angle
        :param tilt: camera pose, tilt angle
        :param f: camera pose, focal length
        """
        self.img = img
        self.img_index = img_index

        """feature points"""

        # a list of key point object (the first return value of detect_compute_sift function)
        self.feature_pts = np.ndarray(0)

        # a [N, 128] int array (the second return value of detect_compute_sift function)
        self.feature_des = np.ndarray(0)

        # a [N] int array of index for keypoint in global_ray
        self.landmark_index = []

        """camera pose"""
        self.pan, self.tilt, self.f = pan, tilt, f

        """camera parameters"""
        # camera center [3] array
        self.center = center
        # rotation matrix [3, 3] array
        self.base_rotation = rotation
        self.u = u
        self.v = v

    def get_feature_num(self):
        """
        :return: keypoint number
        """
        return len(self.feature_pts)

    def convert_keypoint_to_array(self, norm=True):
        N = len(self.feature_pts)
        array_pts = np.zeros((N, 2), dtype=np.float64)
        for i in range(N):
            array_pts[i][0] = self.feature_pts[i].pt[0]
            array_pts[i][1] = self.feature_pts[i].pt[1]

        if norm:
            norm = np.linalg.norm(self.feature_des, axis=1).reshape(-1, 1)
            array_des = np.divide(self.feature_des, norm).astype(np.float64)
        else:
            array_des = self.feature_des.astype(np.float64)

        self.feature_pts = array_pts
        self.feature_des = array_des

    def save_to_mat(self, path):
        """
        save as the format required by random forest.
        :param path: save path for key frame
        """
        keyframe_data = dict()
        keyframe_data['im_name'] = str(self.img_index) + ".jpg"

        # kp, des = detect_compute_sift_array(self.img, 300)

        # kp = add_gauss(kp, 3, 1280, 720)
        # kp = add_outliers(kp, 1, 1280, 720, 40)

        if type(self.feature_pts) == list:
            self.convert_keypoint_to_array()

        keyframe_data['keypoint'] = self.feature_pts
        keyframe_data['descriptor'] = self.feature_des

        # convert the base rotation to (3, 1)
        save_br = np.ndarray([3, 1])
        if self.base_rotation.shape == (3, 3):
            save_br, _ = cv.Rodrigues(self.base_rotation, save_br)
        else:
            save_br = self.base_rotation
        save_br = save_br.ravel()

        keyframe_data['camera'] = np.array([self.u, self.v, self.f, save_br[0], save_br[1], save_br[2],
                                            self.center[0], self.center[1], self.center[2]]).reshape(-1, 1)

        keyframe_data['ptz'] = np.array([self.pan, self.tilt, self.f]).reshape(-1, 1)

        sio.savemat(path, mdict=keyframe_data)
