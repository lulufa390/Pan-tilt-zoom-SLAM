"""
relocalization part is done. @todo: test relocalization when BA is done
"""

import numpy as np
import cv2 as cv
from map import Map
from image_process import *
from sequence_manager import SequenceManager
from transformation import TransFunction


def compute_residual(pose, rays, points, u, v):
    """
    :param pose: shape [3] array of camera pose
    :param rays: [N, 2] array of corresponding rays
    :param points: [N, 2] array of corresponding points
    :param u: camera u
    :param v: camera v
    :return: reprojection error of these points
    """

    residual = np.ndarray([2 * len(rays)])

    for i in range(len(rays)):
        reproject_x, reproject_y = TransFunction.from_pan_tilt_to_2d(u, v, pose[2], pose[0], pose[1], rays[i, 0],
                                                                     rays[i, 1])
        residual[2 * i] = reproject_x - points[i, 0]
        residual[2 * i + 1] = reproject_y - points[i, 1]

    return residual


def relocalization_camera(map, img, pose):
    """
    :param map: object of class Map
    :param img: lost image
    :param pose: lost camera pose: array [3]
    :return: corrected camera pose: array [3]
    """
    kp, des = detect_compute_sift(img, 100)

    nearest_keyframe = 0
    max_matched_num = 0

    matched_kp =[]
    matched_index = []
    for i in range(len(map.keyframe_list)):
        kp_i = map.keyframe_list[i].feature_pts
        des_i = map.keyframe_list[i].feature_des

        kp_inlier, index1, kp_i_inlier, index2 = match_sift_features(kp, des, kp_i, des_i)

        if len(kp_inlier) > max_matched_num:
            max_matched_num = len(kp_inlier)
            nearest_keyframe = i
            matched_kp = kp_inlier
            matched_index = index2

    ray_index = map.keyframe_list[nearest_keyframe].feature_index[matched_index]
    rays = map.global_ray[ray_index]

    u = map.keyframe_list[nearest_keyframe].u
    v = map.keyframe_list[nearest_keyframe].v

    optimized_pose = least_squares(compute_residual, pose, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                                   args=(rays, matched_kp, u, v))

    return optimized_pose


if __name__ == '__main__':

    obj = SequenceManager("./basketball/basketball/basketball_anno.mat",
                          "./basketball/basketball/images",
                          "./objects_basketball.mat")
    relocalization_camera(obj.get_basketball_image_gray(0))
