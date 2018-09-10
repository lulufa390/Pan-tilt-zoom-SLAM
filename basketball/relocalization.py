"""
relocalization part is done. @todo: test relocalization when BA is done
"""

import numpy as np
import cv2 as cv
from map import Map
from image_process import *
from sequence_manager import SequenceManager
from transformation import TransFunction
from key_frame import KeyFrame
from scipy.optimize import least_squares


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

    matched_kp = []
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
    rays = np.array(map.global_ray)[ray_index]

    u = map.keyframe_list[nearest_keyframe].u
    v = map.keyframe_list[nearest_keyframe].v

    optimized_pose = least_squares(compute_residual, pose, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                                   args=(rays, matched_kp, u, v))

    return optimized_pose


if __name__ == '__main__':

    """unit test for relocalization"""
    obj = SequenceManager("./basketball/basketball/basketball_anno.mat",
                          "./basketball/basketball/images",
                          "./objects_basketball.mat")
    img1 = 0
    img2 = 390

    gray1 = obj.get_basketball_image_gray(img1)
    gray2 = obj.get_basketball_image_gray(img2)

    pose1 = obj.get_ptz(img1)
    pose2 = obj.get_ptz(img2)

    mask1 = obj.get_bounding_box_mask(img1)
    mask2 = obj.get_bounding_box_mask(img2)

    keyframe1 = KeyFrame(gray1, img1, obj.c, obj.base_rotation,
                         obj.u, obj.v, pose1[0], pose1[1], pose1[2])
    keyframe2 = KeyFrame(gray2, img2, obj.c, obj.base_rotation,
                         obj.u, obj.v, pose2[0], pose2[1], pose2[2])

    kp1, des1 = detect_compute_sift(gray1, 100)
    after_removed_index1 = remove_player_feature(kp1, mask1)
    kp1 = list(np.array(kp1)[after_removed_index1])
    des1 = des1[after_removed_index1]

    kp2, des2 = detect_compute_sift(gray2, 100)
    after_removed_index2 = remove_player_feature(kp2, mask2)
    kp2 = list(np.array(kp2)[after_removed_index2])
    des2 = des2[after_removed_index2]

    keyframe1.feature_pts = kp1
    keyframe1.feature_des = des1

    keyframe2.feature_pts = kp2
    keyframe2.feature_des = des2

    kp1_inlier, index1, kp2_inlier, index2 = match_sift_features(kp1, des1, kp2, des2)

    cv.imshow("test",
              draw_matches(obj.get_basketball_image(img1), obj.get_basketball_image(img2), kp1_inlier, kp2_inlier))
    cv.waitKey(0)

    map = Map()

    """first frame"""
    for i in range(len(kp1)):
        theta, phi = TransFunction.from_2d_to_pan_tilt(
            obj.u, obj.v, pose1[2], pose1[0], pose1[1], kp1[i].pt[0], kp1[i].pt[1])

        map.global_ray.append(np.array([theta, phi]))

    keyframe1.feature_index = np.array([i for i in range(len(kp1))])

    """second frame"""
    keyframe2.feature_index = np.ndarray([len(kp2)], dtype=np.int32)
    for i in range(len(kp2_inlier)):
        keyframe2.feature_index[index2[i]] = index1[i]

    kp2_outlier_index = list(set([i for i in range(len(des2))]) - set(index2))

    for i in range(len(kp2_outlier_index)):
        theta, phi = TransFunction.from_2d_to_pan_tilt(
            obj.u, obj.v, pose2[2], pose2[0], pose2[1], kp2[kp2_outlier_index[i]].pt[0],
            kp2[kp2_outlier_index[i]].pt[1])
        map.global_ray.append(np.array([theta, phi]))

        keyframe2.feature_index[kp2_outlier_index[i]] = len(map.global_ray) - 1

    map.keyframe_list.append(keyframe1)
    map.keyframe_list.append(keyframe2)

    pose_test = obj.get_ptz(142)

    optimized = relocalization_camera(map=map, img=obj.get_basketball_image_gray(142), pose=np.array([20, -16, 3000]))

    print(pose_test)
    print(optimized.x)
