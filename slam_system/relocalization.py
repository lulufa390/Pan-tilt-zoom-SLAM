"""
Relocalization function.

Function prototype for relocalization:
relocalization_camera(map, img, pose)
return optimized camera pose

Created by Luke, 2018.9
"""

import numpy as np
import cv2 as cv
from scene_map import Map
from image_process import *
from sequence_manager import SequenceManager
from transformation import TransFunction
from key_frame import KeyFrame
from scipy.optimize import least_squares


def _compute_residual(pose, rays, points, u, v):
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
        reproject_x, reproject_y = TransFunction.from_ray_to_image(u, v, pose[2], pose[0], pose[1], rays[i, 0],
                                                                     rays[i, 1])
        residual[2 * i] = reproject_x - points[i, 0]
        residual[2 * i + 1] = reproject_y - points[i, 1]

    return residual


def _recompute_matching_ray(keyframe, img, feature_method):
    """
    :param keyframe: keyframe object to match
    :param img: image to relocalize
    :return: points [N, 2] array in img, rays [N, 2] array in keyframe
    """

    if feature_method == 'sift':
        kp, des = detect_compute_sift(img, 0)
        keyframe_kp, keyframe_des = detect_compute_sift(keyframe.img, 0)
        pt1, index1, pt2, index2 = match_sift_features(kp, des, keyframe_kp, keyframe_des)
    elif feature_method == 'orb':
        kp, des = detect_compute_orb(img, 6000)
        keyframe_kp, keyframe_des = detect_compute_orb(keyframe.img, 6000)
        pt1, index1, pt2, index2 = match_orb_features(kp, des, keyframe_kp, keyframe_des)
    elif feature_method == 'latch':
        kp, des = detect_compute_latch(img, 5000)
        keyframe_kp, keyframe_des = detect_compute_latch(keyframe.img, 5000)
        pt1, index1, pt2, index2 = match_latch_features(kp, des, keyframe_kp, keyframe_des)
    else:
        assert False

    # vis = draw_matches(img, keyframe.img, pt1, pt2)
    # cv.imshow("test", vis)
    # cv.waitKey(0)

    rays = np.ndarray([len(index2), 2])

    for i in range(len(index2)):
        rays[i, 0], rays[i, 1] = TransFunction.from_image_to_ray(keyframe.u, keyframe.v, keyframe.f,
                                                                   keyframe.pan, keyframe.tilt, pt2[i, 0], pt2[i, 1])

    return pt1, rays


def relocalization_camera(map, img, pose):
    """
    :param map: object of class Map
    :param img: lost image
    :param pose: lost camera pose: array [3]
    :return: corrected camera pose: array [3]
    """

    if map.feature_method == 'sift':
        kp, des = detect_compute_sift(img, 0)
    elif map.feature_method == 'orb':
        kp, des = detect_compute_orb(img, 6000)
    elif map.feature_method == 'latch':
        kp, des = detect_compute_latch(img, 5000)
    else:
        assert False

    nearest_keyframe = -1
    max_matched_num = 0

    matched_keyframe_pt = None
    matched_keyframe_index = None
    matched_cur_frame_pt = None

    for i in range(len(map.keyframe_list)):
        keyframe = map.keyframe_list[i]
        # keyframe_kp, keyframe_des = keyframe.feature_pts, keyframe.feature_des

        if map.feature_method == 'sift':
            # keyframe_kp, keyframe_des = keyframe.feature_pts, keyframe.feature_des
            keyframe_kp, keyframe_des = detect_compute_sift(keyframe.img, 1000)
        elif map.feature_method == 'orb':
            keyframe_kp, keyframe_des = detect_compute_orb(keyframe.img, 6000)
        elif map.feature_method == 'latch':
            keyframe_kp, keyframe_des = detect_compute_latch(keyframe.img, 5000)
        else:
            assert False

        if len(keyframe_kp) == 0:
            continue

        print("number", len(keyframe_kp), len(kp))

        if map.feature_method == 'sift':
            pt1, index1, pt2, index2 = match_sift_features(keyframe_kp, keyframe_des, kp, des)
        elif map.feature_method == 'orb':
            pt1, index1, pt2, index2 = match_orb_features(keyframe_kp, keyframe_des, kp, des)
        elif map.feature_method == 'latch':
            pt1, index1, pt2, index2 = match_latch_features(keyframe_kp, keyframe_des, kp, des)
        else:
            assert False

        if index1 is not None:
            if len(index1) > max_matched_num:
                max_matched_num = len(index1)
                nearest_keyframe = i

    # cannot find a good key frame
    if nearest_keyframe == -1:
        print("No matching keyframe!")

        # cv.imwrite("./bundle_result/to_relocalize_frame.jpg", img)
        return pose

    else:
        keyframe = map.keyframe_list[nearest_keyframe]

        points, rays = _recompute_matching_ray(keyframe, img, map.feature_method)

        cv.imwrite("./bundle_result/map_frame.jpg", keyframe.img)
        cv.imwrite("./bundle_result/to_relocalize_frame.jpg", img)

        u = keyframe.u
        v = keyframe.v

        # optimized the camera pose
        optimized_pose = least_squares(_compute_residual, pose, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                                       args=(rays, points, u, v))

        return optimized_pose.x


def ut_relocalization():
    """unit test for relocalization"""
    obj = SequenceManager("../../dataset/basketball/basketball_anno.mat",
                               "../../dataset/basketball/images",
                               "../../dataset/basketball/basketball_ground_truth.mat",
                               "../../dataset/basketball/objects_basketball.mat")
    img1 = 0
    img2 = 390

    gray1 = obj.get_image_gray(img1)
    gray2 = obj.get_image_gray(img2)

    pose1 = obj.get_ptz(img1)
    pose2 = obj.get_ptz(img2)

    mask1 = obj.get_bounding_box_mask(img1)
    mask2 = obj.get_bounding_box_mask(img2)

    camera = obj.get_camera(0)

    keyframe1 = KeyFrame(gray1, img1, camera.camera_center, camera.base_rotation,
                         camera.principal_point[0], camera.principal_point[1], pose1[0], pose1[1], pose1[2])
    keyframe2 = KeyFrame(gray2, img2, camera.camera_center, camera.base_rotation,
                         camera.principal_point[0], camera.principal_point[1], pose2[0], pose2[1], pose2[2])

    kp1, des1 = detect_compute_sift(gray1, 100)
    after_removed_index1 = keypoints_masking(kp1, mask1)
    kp1 = list(np.array(kp1)[after_removed_index1])
    des1 = des1[after_removed_index1]

    kp2, des2 = detect_compute_sift(gray2, 100)
    after_removed_index2 = keypoints_masking(kp2, mask2)
    kp2 = list(np.array(kp2)[after_removed_index2])
    des2 = des2[after_removed_index2]

    keyframe1.feature_pts = kp1
    keyframe1.feature_des = des1

    keyframe2.feature_pts = kp2
    keyframe2.feature_des = des2

    kp1_inlier, index1, kp2_inlier, index2 = match_sift_features(kp1, des1, kp2, des2)

    cv.imshow("test",
              draw_matches(obj.get_image(img1), obj.get_image(img2), kp1_inlier, kp2_inlier))
    cv.waitKey(0)

    map = Map()

    """first frame"""
    for i in range(len(kp1)):
        theta, phi = TransFunction.from_image_to_ray(
            obj.u, obj.v, pose1[2], pose1[0], pose1[1], kp1[i].pt[0], kp1[i].pt[1])

        map.global_ray.append(np.array([theta, phi]))

    keyframe1.landmark_index = np.array([i for i in range(len(kp1))])

    """second frame"""
    keyframe2.landmark_index = np.ndarray([len(kp2)], dtype=np.int32)
    for i in range(len(kp2_inlier)):
        keyframe2.landmark_index[index2[i]] = index1[i]

    kp2_outlier_index = list(set([i for i in range(len(des2))]) - set(index2))

    for i in range(len(kp2_outlier_index)):
        theta, phi = TransFunction.from_image_to_ray(
            obj.u, obj.v, pose2[2], pose2[0], pose2[1], kp2[kp2_outlier_index[i]].pt[0],
            kp2[kp2_outlier_index[i]].pt[1])
        map.global_ray.append(np.array([theta, phi]))

        keyframe2.landmark_index[kp2_outlier_index[i]] = len(map.global_ray) - 1

    map.keyframe_list.append(keyframe1)
    map.keyframe_list.append(keyframe2)

    pose_test = obj.get_ptz(142)

    optimized = relocalization_camera(map=map, img=obj.get_image_gray(142), pose=np.array([20, -16, 3000]))

    print(pose_test)
    print(optimized.x)


if __name__ == '__main__':
    ut_relocalization()
