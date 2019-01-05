"""
This is a Python version to optimize common parameters for broadcast camera model with Levenberg-Marquard algorithm.

Create by Luke, 2019.1
"""

import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
import sys

sys.path.append('../slam_system')
from util import get_projection_matrix_with_camera
from ptz_camera import PTZCamera


def _compute_residual(pose, points3d, points2d, principal_point):
    """
    :param pose: shape [3] array of camera pose
    :param rays: [N, 2] array of corresponding rays
    :param points: [N, 2] array of corresponding points
    :param u: camera u
    :param v: camera v
    :return: reprojection error of these points
    """

    assert len(points2d) == len(points3d)

    camera_num = len(points2d)

    residual_len = 0
    for i in range(camera_num):
        residual_len = residual_len + len(points2d[i])

    residual = np.ndarray([2 * residual_len])

    for i in range(camera_num):
        for j in range(len(points2d[i])):
            base_rotation = np.ndarray([3, 3])
            cv.Rodrigues(pose[3:6], base_rotation)

            print(base_rotation)
            print("pose", pose)

            broadcast_camera = PTZCamera(principal_point, pose[0:3], base_rotation, pose[6:12])

            reproject_x, reproject_y = broadcast_camera.project_3d_point(points3d[i][j])

            residual[2 * i] = reproject_x - points2d[i, 0]
            residual[2 * i + 1] = reproject_y - points2d[i, 1]

    return residual


def optimize_broadcast_camera_parameter(model_3d_points, init_cameras, init_rod):
    """
    optimize broadcast camera parameters using LMQ mehtod
    It is slow
    :param model_3d_points: N * 3, N is the number of 3d points
    :param init_cameras:  M * 9, M is the number of camera poses
    :param init_rod:  3 * 1
    :return: N * 9 cameras, N * 3 pan-tilt-zooms,
        12 * 1 shared parameters, camera_center, base rotation and lambda
    """
    assert model_3d_points.shape[1] == 3
    assert init_cameras.shape[1] == 9
    assert init_rod.shape[0] == 3

    point_num = model_3d_points.shape[0]
    camera_num = init_cameras.shape[0]

    image_width = init_cameras[0, 0] * 2
    image_height = init_cameras[0, 1] * 2

    principal_point = init_cameras[0, 0:2]

    # prepare points data for both 3d points and 2d points on images
    points3d = [[] for _ in range(camera_num)]
    points2d = [[] for _ in range(camera_num)]

    for i in range(camera_num):
        matrix = get_projection_matrix_with_camera(init_cameras[i])
        for j in range(point_num):
            point3d = model_3d_points[j]
            p_homo = np.array([point3d[0], point3d[1], point3d[2], 1])
            uvw = np.dot(matrix, p_homo)  # 3_4 * 4_1
            assert uvw[2] != 0.0
            point2d = np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]])

            if 0 < point2d[0] < image_width and 0 < point2d[1] < image_height:
                points3d[i].append(point3d)
                points2d[i].append(point2d)

    pose = np.zeros(camera_num * 3 + 12)

    pose[-12:-9] = init_cameras[0, 6:9]
    pose[-9:-6] = init_rod.ravel()


    # use least square
    optimized_pose = least_squares(_compute_residual, pose, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                                   args=(points3d, points2d, principal_point))

    opt_ptzs = optimized_pose.x[:-12].reshape([-1, 3])

    shared_parameters = optimized_pose.x[-12:]

    opt_cameras = np.zeros((camera_num, 9))

    for i in range(camera_num):
        opt_cameras[i, 0:2] = principal_point
        opt_cameras[i, 3] = opt_ptzs[i, 2]

        # how to get this
        opt_cameras[i, 3:6] = np.zeros(3)

        opt_cameras[i, 6:9] = shared_parameters[0:3]

    return opt_cameras, opt_ptzs, shared_parameters


if __name__ == "__main__":
    print("test")
