"""
Bundle Adjustment function
"""

import scipy.io as sio
import cv2 as cv
import numpy as np
import math
from scipy.optimize import least_squares

from key_frame import KeyFrame
from image_process import build_matching_graph, draw_matches
from sequence_manager import SequenceManager
from transformation import TransFunction
#from scene_map import Map
from util import overlap_pan_angle
import random



def _compute_residual(x, n_pose, n_landmark, n_residual, keypoints, src_pt_index, dst_pt_index, landmark_index, u, v, reference_pose, verbose = False):
    """
    The function compute residuals form N - 1 camera poses and M landmarks
    :param x: N - 1 * 3 camera pose, pan, tilt, focal_length + M * 2 landmark, (pan, tilt)
    :n_pose: camera pose number, only N - 1 poses are actually optimized
    :n_landmark: landmark number
    :n_residual: number of residuals
    :param keypoints: list of N*2 matrix, (x, y) in each frame
    : src_pt_index, dst_pt_index, landmark_index: 2D list of indices
    :u, v: image center
    :reference_pose: camera pose of the reference frame. This frame must be the first frame
    :return: residual
    """
    # check input dimension
    assert x.shape[0] == (n_pose - 1) * 3 + n_landmark * 2
    N = len(keypoints)
    assert n_pose == N
    assert len(src_pt_index) == N
    assert len(dst_pt_index) == N
    assert len(landmark_index) == N
    assert reference_pose.shape[0] == 3

    for i in range(N):
        assert len(src_pt_index[i]) == N
        assert len(dst_pt_index[i]) == N
        assert len(landmark_index[i]) == N

    landmark_start_index = n_pose * 3

    # step 1: prepare data
    # add the reference pose as the first camera pose, because x only has n_pose - 1 poses
    x0 = np.zeros([n_pose*3 + n_landmark*2])
    x0[0:3] = reference_pose  # first pose
    x0[3:] = x                # the rest pose and landmarks

    residual = np.ndarray(n_residual)
    residual_idx = 0

    reprojection_err = 0.0
    # step 2: compute residual
    # loop each image pairs
    for i in range(N):
        for j in range(N):
            pan1, tilt1, fl1 = x0[i * 3 + 0], x0[i * 3 + 1], x0[i * 3 + 2]  # estimated camera pose
            pan2, tilt2, fl2 = x0[j * 3 + 0], x0[j * 3 + 1], x0[j * 3 + 2]

            # loop each keypoint matches from image i to image j
            for idx1, idx2, idx3 in zip(src_pt_index[i][j], dst_pt_index[i][j], landmark_index[i][j]):
                pan, tilt = x0[landmark_start_index + idx3 * 2], x0[landmark_start_index + idx3 * 2 + 1]  # estimated landmark pan, tilt
                x1, y1 = keypoints[i][idx1][0], keypoints[i][idx1][1]  # observation
                x2, y2 = keypoints[j][idx2][0], keypoints[j][idx2][1]
                proj_x1, proj_y1 = TransFunction.from_pan_tilt_to_2d(u, v, fl1, pan1, tilt1, pan, tilt)  # projection by pose and landmark
                proj_x2, proj_y2 = TransFunction.from_pan_tilt_to_2d(u, v, fl2, pan2, tilt2, pan, tilt)

                # point in camera i
                dx = proj_x1 - x1
                dy = proj_y1 - y1

                residual[residual_idx] = dx
                residual_idx += 1
                residual[residual_idx] = dy
                residual_idx += 1
                reprojection_err += math.sqrt(dx*dx + dy*dy)

                # point in camera j
                dx = proj_x2 - x2
                dy = proj_y2 - y2
                residual[residual_idx] = dx
                residual_idx += 1
                residual[residual_idx] = dy
                residual_idx += 1
                reprojection_err += math.sqrt(dx * dx + dy * dy)

    assert residual_idx == n_residual

    # debug
    if verbose:
        print("reprojection error is %f" % (reprojection_err/(n_residual/2)))
    return residual

def bundle_adjustment(images, image_indices, initial_ptzs, center, rotation, u, v, save_path, verbose = False):
    """
    build a map from image matching: it takes long time
    assumption: first camera pose is the ground truth
    :param images: a list of images
    :param image_indices: a list of image indices
    :param initial_ptzs: N * 3, pan, tilt, focal_length
    :param center:
    :param rotation: base rotation
    :param u:
    :param v:
    :param save_path: a path to save pair-wise image matching
    :return: a map
    """
    # check input parameters
    N = len(images)
    assert N >= 1
    assert len(image_indices) == N
    assert initial_ptzs.shape[0] == N and initial_ptzs.shape[1] == 3
    assert center.shape[0] == 3 and rotation.shape[0] == 3 and rotation.shape[1] == 3

    # step 1: image matching
    # initial image matching
    image_match_mask = [[0 for i in range(N)] for i in range(N)]
    overlap_angle_threshold = 5  # degrees
    for i in range(N):
        pan1, fl1 = initial_ptzs[i][0], initial_ptzs[i][2]
        for j in range(N):
            pan2, fl2 = initial_ptzs[j][0], initial_ptzs[j][2]
            angle = overlap_pan_angle(fl1, pan1, fl2, pan2, 1280)
            # print('overlap pan angle of (%d %d) is %d'% (i, j, angle))
            if angle > overlap_angle_threshold:
                image_match_mask[i][j] = 1

    # matching keypoints in images
    keypoints, descriptors, points, src_pt_index, dst_pt_index, landmark_index, n_landmark = build_matching_graph(images, image_match_mask, verbose)

    # save image matching result for debug
    for i in range(N):
        for j in range(N):
            if len(src_pt_index[i][j]) != 0:
                pts1 = points[i].take(src_pt_index[i][j], axis=0)
                pts2 = points[j].take(dst_pt_index[i][j], axis=0)
                vis = draw_matches(images[i], images[j], pts1, pts2)
                save_name = save_path + '/' + str(i) + '_' + str(j) + '.jpg'
                cv.imwrite(save_name, vis)
                print('save matching result to: %s' % save_name)
                #cv.imshow('matching result', vis)
                #cv.waitKey(0)
    if verbose:
        print('Complete pair-wise image matching')

    # step 2: prepare bundle adjustment data
    n_residual = 0
    for i in range(N):
        for j in range(N):
            n_residual += len(src_pt_index[i][j]) * 4
    print('residual number is %d.' % n_residual)

    # initialize reference camera pose
    ref_pose = initial_ptzs[0]

    # initial value of pose and rays
    x0 = np.zeros([N * 3 + n_landmark * 2])
    for i in range(N):
        ptz = initial_ptzs[i]
        x0[i*3: i*3 + 3] = ptz

    landmark_start_index = N * 3
    for i in range(N):
        ptz1 = x0[i * 3: i * 3 + 3]
        for j in range(N):
            ptz2 = x0[j * 3: j * 3 + 3]
            for idx1, idx2, idx3 in zip(src_pt_index[i][j], dst_pt_index[i][j], landmark_index[i][j]):
                x1, y1 = points[i][idx1][0], points[i][idx1][1]
                x2, y2 = points[j][idx2][0], points[j][idx2][1]

                #@tod landmark initialization. may overwrite the previously inializaed landmarks
                landmark = TransFunction.from_2d_to_pan_tilt(u, v, ptz1[2], ptz1[0], ptz1[1], x1, y1)
                x0[landmark_start_index + idx3 * 2: landmark_start_index + idx3 * 2 + 2] = landmark
    n_pose = N

    x0 = x0[3:]  # remove first camera pose so that it is not optimized


    # step 3: camera pose and landmark optimization
    optimized = least_squares(_compute_residual, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                              args=(n_pose, n_landmark, n_residual, points, src_pt_index, dst_pt_index,
                                    landmark_index, u, v, ref_pose))

    optimized_pose = optimized.x[0:landmark_start_index - 3]
    all_poses = np.zeros((n_pose * 3))
    all_poses[0:3] = ref_pose
    all_poses[3:] = optimized_pose
    optimized_landmarks = optimized.x[landmark_start_index - 3:]

    # step 4: check reprojection error @todo
    if verbose:
        print("@todo: check reprojectoin error")

    # step 5: write optimization result to a map: N*2 landmarks and a list of keyframes
    optimized_landmarks = optimized_landmarks.reshape(-1, 2)
    keyframes = []
    for i in range(N):
        pan, tilt, fl, = all_poses[i * 3:i * 3 + 3]
        key_frame = KeyFrame(images[i], image_indices[i], center, rotation, u, v, pan, tilt, fl)

        # collect all (local, global) pairs from image i
        pairs = []
        for j in range(N):
            if len(src_pt_index[i][j]) == 0:
                continue
            for idx1, idx3 in zip(src_pt_index[i][j], landmark_index[i][j]):
                pairs.append((idx1, idx3))
        pairs = set(pairs)  # remove redundant
        local_index, global_index = [], []
        for pair in pairs:
            local_index.append(pair[0])
            global_index.append(pair[1])

        # save result to key frame
        key_frame.feature_pts = [keypoints[i][j] for j in local_index]
        key_frame.feature_des = descriptors[i].take(local_index, axis=0)
        key_frame.landmark_index = np.array(global_index, dtype=np.int32)
        keyframes.append(key_frame)

    # step 6:
    return optimized_landmarks, keyframes



def ut_build_adjustment_from_image_sequence():
    input = SequenceManager("/Users/jimmy/Desktop/ptz_slam_dataset/basketball/basketball_anno.mat",
                            "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images",
                            "/Users/jimmy/PycharmProjects/ptz_slam/Camera-Calibration/basketball/objects_basketball.mat")

    cc = input.get_camera_center()
    base_rotation = input.get_base_rotation()
    u = 1280/2
    v = 720/2

    image_index = [0, 660, 680] #680, 690, 700, 730, 800

    N = len(image_index)
    key_frames = []
    # initialize key frames
    for i in range(len(image_index)):
        im = input.get_basketball_image(image_index[i])
        ptz = input.get_ptz(image_index[i])
        key_frame = KeyFrame(im, i, cc, base_rotation, u, v, ptz[0], ptz[1], ptz[2])
        key_frames.append(key_frame)
        #print(ptz)

    # step 1: rough matching pairs
    images = [key_frames[i].img for i in range(len(key_frames))]
    image_match_mask = [[0 for i in range(N)] for i in range(N)]
    overlap_angle_threshold = 5
    for i in range(N):
        pan1, fl1 = key_frames[i].pan, key_frames[i].f
        for j in range(N):
            pan2, fl2 = key_frames[j].pan, key_frames[j].f
            angle = _overlap_pan_angle(fl1, pan1, fl2, pan2, 1280)
            #print('overlap pan angle of (%d %d) is %d'% (i, j, angle))
            if angle > overlap_angle_threshold:
                image_match_mask[i][j] = 1

    # step 2: keypoint matching in image pairs
    # matching keypoints in images
    keypoints, descriptors, points, src_pt_index, dst_pt_index, landmark_index, n_landmark = build_matching_graph(images, image_match_mask, True)

    # check landmark index
    if 0:
        for i in range(N):
            for j in range(N):
                if len(src_pt_index[i][j]) == 0:
                    continue
                print('landmark, source, destination: %d %d' % (i, j))
                for idx1, idx2, idx3 in zip(src_pt_index[i][j], dst_pt_index[i][j], landmark_index[i][j]):
                    print(idx3, idx1, idx2)
                print('end.......................')

    if 0:
        # test image matching result
        for i in range(N):
            for j in range(N):
                if len(src_pt_index[i][j]) != 0:
                    pts1 = points[i].take(src_pt_index[i][j], axis = 0)
                    pts2 = points[j].take(dst_pt_index[i][j], axis = 0)
                    vis = draw_matches(images[i], images[j], pts1, pts2)
                    cv.imshow('matching result', vis)
                    cv.waitKey(0)

    # prepare data from least square optimization
    # def compute_residual(x, n_pose, n_landmark, n_residual, keypoints, src_pt_index, dst_pt_index, landmark_index, u, v):
    n_residual = 0
    for i in range(N):
        for j in range(N):
            n_residual += len(src_pt_index[i][j]) * 4
    print('residual number is %d.' % n_residual)

    # initialize reference camera pose
    ref_pose = input.get_ptz(image_index[0])



    # initial value of pose and rays
    pose_gt = np.zeros((N*3))
    x0 = np.zeros([N*3 + n_landmark*2])
    for i in range(len(image_index)):
        ptz = input.get_ptz(image_index[i])
        x0[i * 3 + 0], x0[i * 3 + 1], x0[i * 3 + 2] = ptz
        if i != 0:
            x0[i * 3 + 0] += random.gauss(0, 1)
            x0[i * 3 + 2] += random.gauss(0, 50)

        pose_gt[i*3:i*3+3] = ptz

    landmark_start_index = N * 3
    for i in range(N):
        ptz1 = x0[i*3: i*3+3]
        for j in range(N):
            ptz2 = x0[j*3 : j*3+3]
            for idx1, idx2, idx3 in zip(src_pt_index[i][j], dst_pt_index[i][j], landmark_index[i][j]):
                x1, y1 = points[i][idx1][0], points[i][idx1][1]
                x2, y2 = points[j][idx2][0], points[j][idx2][1]

                x1 += random.gauss(0, 1)
                y1 += random.gauss(0, 1)
                landmark = TransFunction.from_2d_to_pan_tilt(u, v, ptz1[2], ptz1[0], ptz1[1], x1, y1)
                x0[landmark_start_index + idx3*2: landmark_start_index + idx3*2 + 2] = landmark
    n_pose = N

    x0 = x0[3:] # remove first camera pose

    # optimize camera pose and landmark
    optimized = least_squares(compute_residual, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                                   args=(n_pose, n_landmark, n_residual, points, src_pt_index, dst_pt_index, landmark_index, u, v, ref_pose))


    optimized_pose = optimized.x[0:landmark_start_index-3]
    pose_estimate = np.zeros((n_pose*3))
    pose_estimate[0:3] = ref_pose
    pose_estimate[3:] = optimized_pose

    optimized_landmarks = optimized.x[landmark_start_index-3:]

    dif = pose_gt - pose_estimate
    dif = dif.reshape((-1, 3))
    print(dif)

    # store result to a map
    a_map = Map()
    a_map.global_ray = optimized_landmarks.reshape(-1, 2)
    for i in range(N):
        pan, tilt, fl, = pose_estimate[i*3:i*3+3]
        keyframe = KeyFrame(images[i], image_index[i], cc, base_rotation, u, v, pan, tilt, fl)

        # collect all (local, global) pairs from image i
        pairs = []
        for j in range(N):
            if len(src_pt_index[i][j]) == 0:
                continue
            for idx1, idx3 in zip(src_pt_index[i][j], landmark_index[i][j]):
                pairs.append((idx1, idx3))
        pairs = set(pairs)  # remove redundant
        local_index, global_index = [], []
        for pair in pairs:
            local_index.append(pair[0])
            global_index.append(pair[1])

        # save result to key frame
        key_frame.feature_pts = [keypoints[i][j] for j in local_index]
        key_frame.feature_des = descriptors[i].take(local_index, axis = 0)
        key_frame.landmark_index = np.array(global_index, dtype=np.int32)
        a_map.keyframe_list.append(key_frame)


def ut_bundle_adjustment_interface():
    #def bundle_adjustment(images, image_indices, initial_ptzs, center, rotation, u, v, save_path, verbose=False):

    input = SequenceManager("/Users/jimmy/Desktop/ptz_slam_dataset/basketball/basketball_anno.mat",
                            "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images",
                            "/Users/jimmy/PycharmProjects/ptz_slam/Camera-Calibration/basketball/objects_basketball.mat")

    camera_center = input.get_camera_center()
    base_rotation = input.get_base_rotation()
    u = 1280 / 2
    v = 720 / 2

    image_index = [0, 660, 680]  # 680, 690, 700, 730, 800

    N = len(image_index)
    initial_ptzs = np.zeros((N, 3))
    images = []
    for i in range(len(image_index)):
        im = input.get_basketball_image(image_index[i])
        ptz = input.get_ptz(image_index[i])
        # add noise to the rest of cameras
        if i != 0:
            ptz[0] += random.gauss(0, 1)
            ptz[2] += random.gauss(0, 50)
        initial_ptzs[i] = ptz
        images.append(im)


    landmarks, keyframes = bundle_adjustment(images, image_index, initial_ptzs, camera_center, base_rotation, u, v, '.', True)




def ut_least_square():
    def fun_rosenbrock(x):
        return np.array([10 * (x[1] - x[0]**2) , (1 - x[0])])

    from scipy.optimize import least_squares
    x0 = np.array([2, 2])
    res = least_squares(fun_rosenbrock, x0)
    print(res.x)


if __name__ == '__main__':
    #ut_build_adjustment_from_image_sequence()
    ut_bundle_adjustment_interface()

    #ut_least_square()
