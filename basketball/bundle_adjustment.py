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


# internal class
class _FrameToFrameMatch:
    def __init__(self, src_index, des_index, src_pts_index, des_pts_index, landmark_index):
        self.src_index = src_index
        self.des_index = des_index

        self.src_pts_index = src_pts_index
        self.des_pts_index = des_pts_index
        self.landmark_index = landmark_index


def compute_residual(x, n_pose, n_landmark, n_residual, keypoints, src_pt_index, dst_pt_index, landmark_index, u, v, reference_pose):
    """
    :param x: N * 3 camera pose, pan, tilt, focal_length + M * 2 landmark, (pan, tilt)
    :n_pose: camera pose number
    :n_landmark: landmark number
    :n_residual: number of residuals
    :param keypoints: list of N*2 matrix, (x, y) in each frame
    : src_pt_index, dst_pt_index, landmark_index: 2D list of indices
    :u, v: image center
    :return: residual
    """
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

    # x --> only has "n_pose - 1" poses
    x0 = np.zeros([n_pose*3 + n_landmark*2])
    x0[0:3] = reference_pose  # first pose
    x0[3:] = x                # the rest pose and landmarks

    residual = np.ndarray(n_residual)
    residual_idx = 0

    reprojection_err = 0.0
    for i in range(N):
        for j in range(N):
            pan1, tilt1, fl1 = x0[i * 3 + 0], x0[i * 3 + 1], x0[i * 3 + 2]  # camera pose
            pan2, tilt2, fl2 = x0[j * 3 + 0], x0[j * 3 + 1], x0[j * 3 + 2]

            # keypoint index
            for idx1, idx2, idx3 in zip(src_pt_index[i][j], dst_pt_index[i][j], landmark_index[i][j]):
                pan, tilt = x0[landmark_start_index + idx3 * 2], x0[landmark_start_index + idx3 * 2 + 1]  # landmark pan, tilt
                x1, y1 = keypoints[i][idx1][0], keypoints[i][idx1][1]
                x2, y2 = keypoints[j][idx2][0], keypoints[j][idx2][1]
                proj_x1, proj_y1 = TransFunction.from_pan_tilt_to_2d(u, v, fl1, pan1, tilt1, pan, tilt)
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

    #print("reprojection error is %f" % (reprojection_err/(n_residual/2)))

    return residual





def ut_test_build_adjustment():

    def overlap_pan_angle(fl_1, pan_1, fl_2, pan_2, im_width):
        # overlap angle (in degree) between two cameras

        w = im_width/2
        delta_angle = math.atan(w/fl_1) * 180.0/math.pi
        pan1_min = pan_1 - delta_angle
        pan1_max = pan_1 + delta_angle


        delta_angle = math.atan(w/fl_2) * 180.0/math.pi
        pan2_min = pan_2 - delta_angle
        pan2_max = pan_2 + delta_angle

        angle1 = max(pan1_min, pan2_min)
        angle2 = min(pan1_max, pan2_max)
        #print(fl1, pan1, fl2, pan2, angle1, angle2)

        return max(0, angle2 - angle1)



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
            angle = overlap_pan_angle(fl1, pan1, fl2, pan2, 1280)
            #print('overlap pan angle of (%d %d) is %d'% (i, j, angle))
            if angle > overlap_angle_threshold:
                image_match_mask[i][j] = 1

    # step 2: keypoint matching in image pairs
    # matching keypoints in images
    keypoints, descriptors, src_pt_index, dst_pt_index, landmark_index, n_landmark = build_matching_graph(images, image_match_mask, True)

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
                    pts1 = keypoints[i].take(src_pt_index[i][j], axis = 0)
                    pts2 = keypoints[j].take(dst_pt_index[i][j], axis = 0)
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


    import random
    # initial value of pose and rays
    pose_gt = np.zeros((N*3))
    x0 = np.zeros([N*3 + n_landmark*2])
    for i in range(len(image_index)):
        ptz = input.get_ptz(image_index[i])
        x0[i * 3 + 0], x0[i * 3 + 1], x0[i * 3 + 2] = ptz
        if i != 0:
            x0[i * 3 + 2] += random.gauss(0, 10)

        pose_gt[i*3:i*3+3] = ptz

    # def from_2d_to_pan_tilt(u, v, f, c_p, c_t, x, y):
    landmark_start_index = N * 3
    for i in range(N):
        ptz1 = x0[i*3: i*3+3]
        for j in range(N):
            ptz2 = x0[j*3 : j*3+3]
            for idx1, idx2, idx3 in zip(src_pt_index[i][j], dst_pt_index[i][j], landmark_index[i][j]):
                x1, y1 = keypoints[i][idx1][0], keypoints[i][idx1][1]
                x2, y2 = keypoints[j][idx2][0], keypoints[j][idx2][1]

                x1 += random.gauss(0, 1)
                y1 += random.gauss(0, 1)
                landmark = TransFunction.from_2d_to_pan_tilt(u, v, ptz1[2], ptz1[0], ptz1[1], x1, y1)
                x0[landmark_start_index + idx3*2: landmark_start_index + idx3*2 + 2] = landmark
    n_pose = N

    x0 = x0[3:] # remove first camera pose
    optimized = least_squares(compute_residual, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                                   args=(n_pose, n_landmark, n_residual, keypoints, src_pt_index, dst_pt_index, landmark_index, u, v, ref_pose))


    pose_estimate = optimized.x[0:landmark_start_index-3]
    pose_gt = pose_gt[3:]  # remove the first camera pose

    dif = pose_gt - pose_estimate
    dif = dif.reshape((-1, 3))
    print(dif)















def ut_least_square():

    def fun_rosenbrock(x):
        return np.array([10 * (x[1] - x[0]**2) , (1 - x[0])])

    from scipy.optimize import least_squares
    x0 = np.array([2, 2])
    res = least_squares(fun_rosenbrock, x0)
    print(res.x)


if __name__ == '__main__':
    ut_test_build_adjustment()

    #ut_least_square()
