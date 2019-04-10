"""
Keyframe class.

Created by Luke, 2018.9
"""

import scipy.io as sio
import numpy as np
import cv2 as cv
from image_process import detect_compute_sift_array


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
        self.feature_pts = []

        # a [N, 128] int array (the second return value of detect_compute_sift function)
        self.feature_des = []

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

    def save_to_mat(self, path):
        """
        save as the format required by random forest.
        :param path: save path for key frame
        """
        keyframe_data = dict()
        keyframe_data['im_name'] = str(self.img_index) + ".jpg"

        kp, des = detect_compute_sift_array(self.img, 1500)

        keyframe_data['keypoint'] = kp
        keyframe_data['descriptor'] = des

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
