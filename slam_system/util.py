"""
Other functions.

Create by Jimmy, 2018.9
"""

import math
import numpy as np
import scipy.io as sio
import cv2 as cv
import random

from sys import platform as sys_pf

import matplotlib

if sys_pf == 'darwin':
    matplotlib.use("TkAgg")
elif sys_pf == 'win32':
    matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt

from image_process import blur_sub_image


def get_projection_matrix_with_camera(camera):
    """
    :param camera: len = 9 array(principle point, f, Rx, Ry, Rz, Cx, Cy, Cz)
    :return: 3 * 4 projection matrix
    """
    k = np.array([[camera[2], 0, camera[0]],
                  [0, camera[2], camera[1]],
                  [0, 0, 1]])

    rod = camera[3:6]
    rotation = np.zeros((3, 3))
    cv.Rodrigues(rod, rotation)

    cc = np.eye(3, 4)
    cc[0][3] = -camera[6]
    cc[1][3] = -camera[7]
    cc[2][3] = -camera[8]

    project_mat = np.dot(k, np.dot(rotation, cc))
    return project_mat


def overlap_pan_angle(fl_1, pan_1, fl_2, pan_2, im_width):
    """
    :param fl_1: focal length in pixel
    :param pan_1:  pan angle in degree
    :param fl_2:
    :param pan_2:
    :param im_width: image with in pixel
    :return: overlapped pan angle
    """
    # overlap angle (in degree) between two cameras

    w = im_width / 2
    delta_angle = math.atan(w / fl_1) * 180.0 / math.pi
    pan1_min = pan_1 - delta_angle
    pan1_max = pan_1 + delta_angle

    delta_angle = math.atan(w / fl_2) * 180.0 / math.pi
    pan2_min = pan_2 - delta_angle
    pan2_max = pan_2 + delta_angle

    angle1 = max(pan1_min, pan2_min)
    angle2 = min(pan1_max, pan2_max)

    return max(0, angle2 - angle1)


def get_overlap_index(index1, index2):
    """
    This function get two arrays, and return the shared numbers in these two arrays as an new array.
    :param index1: array 1
    :param index2: array 2
    :return: overlapped numbers in two array
    """
    index1_overlap = np.ndarray([0], np.int8)
    index2_overlap = np.ndarray([0], np.int8)
    ptr1 = 0
    ptr2 = 0
    while ptr1 < len(index1) and ptr2 < len(index2):
        if index1[ptr1] == index2[ptr2]:
            index1_overlap = np.append(index1_overlap, ptr1)
            index2_overlap = np.append(index2_overlap, ptr2)
            ptr1 += 1
            ptr2 += 1
        elif index1[ptr1] < index2[ptr2]:
            ptr1 += 1
        elif index1[ptr1] > index2[ptr2]:
            ptr2 += 1
    return index1_overlap, index2_overlap


def add_gauss(points, var, max_width, max_height):
    """
    Add Gaussian noise to 2D points (NumPy ndarray).
    :param points: array [N, 2]
    :param var: variance for Gauss distribution
    :return: array [N, 2] with noise
    """
    noise_points = np.zeros_like(points)
    for i in range(len(points)):
        noise_points[i, 0] = points[i, 0] + random.gauss(0, var)
        noise_points[i, 1] = points[i, 1] + random.gauss(0, var)

        if noise_points[i, 0] >= max_width:
            noise_points[i, 0] = max_width - 1

        if noise_points[i, 0] < 0:
            noise_points[i, 0] = 0

        if noise_points[i, 1] >= max_height:
            noise_points[i, 1] = max_height - 1

        if noise_points[i, 1] < 0:
            noise_points[i, 1] = 0

    return noise_points


def add_outliers(points, var, max_width, max_height, percentage):
    pts_with_outliers = points.copy()
    N = points.shape[0]

    sample_list = [i for i in range(N)]
    sample_list = random.sample(sample_list, int(percentage / 100 * len(sample_list)))

    for i in sample_list:
        pts_with_outliers[i, 0] = random.uniform(0, max_width - 1)
        pts_with_outliers[i, 1] = random.uniform(0, max_height - 1)

    output = add_gauss(pts_with_outliers, var, max_width, max_height)

    return output


def add_gauss_cv_keypoints(points, var, max_width, max_height):
    """
    Add Gaussian noise to 2D points (OpenCV keypoints type).
    :param points: list of OpenCV keypoints
    :param var: variance for Gauss distribution
    :return: list of OpenCV keypoints with noise
    """
    for i in range(len(points)):
        new_x = points[i].pt[0] + random.gauss(0, var)
        new_y = points[i].pt[1] + random.gauss(0, var)

        if new_x >= max_width:
            new_x = max_width - 1

        if new_x < 0:
            new_x = 0

        if new_y >= max_height:
            new_y = max_height - 1

        if new_y < 0:
            new_y = 0
        points[i].pt = (new_x, new_y)
    return points


def add_outliers_cv_keypoints(points, var, max_width, max_height, percentage):
    pts_with_outliers = points.copy()
    N = len(points)

    sample_list = [i for i in range(N)]
    sample_list = random.sample(sample_list, int(percentage / 100 * len(sample_list)))

    for i in sample_list:
        new_x = random.uniform(0, max_width - 1)
        new_y = random.uniform(0, max_height - 1)

        pts_with_outliers[i].pt = (new_x, new_y)

    output = add_gauss_cv_keypoints(pts_with_outliers, var, max_width, max_height)

    return output

def uniform_point_sample_on_field(x_max, y_max, x_num, y_num):
    """
    Uniformly get point samples on play field.
    the origin point is the left-down corner.
    :param x_max: the max x of point (field length)
    :param y_max: the max y of point (field width)
    :param x_num: number of points on x direction
    :param y_num: number of points on y direction
    :return: ndarray of [N, 2], N is determined by length(width) and its step.
    """

    point_list = []

    for x in np.linspace(0, x_max, x_num):
        for y in np.linspace(0, y_max, y_num):
            point_list.append([x, y, 0])

    return np.array(point_list)


def draw_camera_plot(ground_truth_pan, ground_truth_tilt, ground_truth_f,
                     estimate_pan, estimate_tilt, estimate_f):
    """
    draw plot for ground truth and estimated camera pose.
    """

    sequence_length = len(ground_truth_pan)

    plt.figure("pan percentage error")
    x = np.array([i for i in range(sequence_length)])
    plt.plot(x, (estimate_pan - ground_truth_pan) / ground_truth_pan * 100, 'b', label='predict')
    plt.xlabel("frame")
    plt.ylabel("error %")
    plt.legend(loc="best")

    plt.figure("tilt percentage error")
    x = np.array([i for i in range(sequence_length)])
    plt.plot(x, (estimate_tilt - ground_truth_tilt) / ground_truth_tilt * 100, 'b', label='predict')
    plt.xlabel("frame")
    plt.ylabel("error %")
    plt.legend(loc="best")

    plt.figure("f percentage error")
    x = np.array([i for i in range(sequence_length)])
    plt.plot(x, (estimate_f - ground_truth_f) / ground_truth_f * 100, 'b', label='predict')
    plt.xlabel("frame")
    plt.ylabel("error %")
    plt.legend(loc="best")

    """absolute value"""
    plt.figure("pan")
    x = np.array([i for i in range(sequence_length)])
    plt.plot(x, ground_truth_pan, 'r', label='ground truth')
    plt.plot(x, estimate_pan, 'b', label='predict')
    plt.xlabel("frame")
    plt.ylabel("pan angle")
    plt.legend(loc="best")

    plt.figure("tilt")
    x = np.array([i for i in range(sequence_length)])
    plt.plot(x, ground_truth_tilt, 'r', label='ground truth')
    plt.plot(x, estimate_tilt, 'b', label='predict')
    plt.xlabel("frame")
    plt.ylabel("tilt angle")
    plt.legend(loc="best")

    plt.figure("f")
    x = np.array([i for i in range(sequence_length)])
    plt.plot(x, ground_truth_f, 'r', label='ground truth')
    plt.plot(x, estimate_f, 'b', label='predict')
    plt.xlabel("frame")
    plt.ylabel("f")
    plt.legend(loc="best")

    plt.show()


def save_camera_pose(pan, tilt, f, path):
    """
    This function saves camera pose to .mat file.
    Assume the length of sequence is n.
    :param pan: an array [n] of pan angle
    :param tilt: an array [n] of tilt angle
    :param f: an array [n] of focal length
    :param path: folder path for mat file
    """
    camera_pose = dict()
    camera_pose['pan'] = pan
    camera_pose['tilt'] = tilt
    camera_pose['f'] = f

    sio.savemat(path, mdict=camera_pose)


def load_camera_pose(path, separate=False):
    """
    :param path: file path for .mat
    :param separate: the pan-tilt-zoom pose saved in one array or separate(3) arrays
    :return: 3 arrays (pan, tilt, zoom) each of size [n]. (n is length of sequence)
    """

    camera_pos = sio.loadmat(path)

    if separate:
        pan = camera_pos['pan'].squeeze()
        tilt = camera_pos['tilt'].squeeze()
        focal_length = camera_pos['f'].squeeze()

    else:
        ptz = camera_pos['ptz']
        pan, tilt, focal_length = ptz[:, 0], ptz[:, 1], ptz[:, 2]

    return pan, tilt, focal_length


def compute_error_data(ptz, ground_truth):
    """
    :param ptz: tuple (3) of pan, tilt, f array
    :param ground_truth: tuple (3) of pan, tilt, f array (gt)
    :return: mean errors
    """

    test_p, test_t, test_f = ptz
    gt_p, gt_t, gt_f = ground_truth

    error_p = np.mean(np.fabs(test_p - gt_p))
    error_t = np.mean(np.fabs(test_t - gt_t))
    error_f = np.mean(np.fabs(test_f - gt_f))

    return error_p, error_t, error_f


def video_capture(file_path, save_path, begin_time, rate, length):
    """
    function to capture video into images.
    :param file_path: video path
    :param save_path: image folder
    :param begin_time: begin time slice
    :param rate: rate for video
    :param length: length in frame number
    """
    video = cv.VideoCapture(file_path)

    for i in range(0, length):
        print(i)
        video.set(cv.CAP_PROP_POS_MSEC, begin_time + i / rate * 1000)
        _, img = video.read()

        # img = blur_sub_image(img, 61, 32, 564, 46)
        # img = blur_sub_image(img, 1100, 30, 85, 76)

        # img = blur_sub_image(img, 227, 55, 130, 100)
        img = blur_sub_image(img, 208, 31, 171, 100)

        # cv.imshow("test", img)

        cv.imwrite(save_path + "/" + str(i + 1) + ".jpg", img)

        # cv.waitKey(0)


def ut_add_gaussian():
    import numpy as np
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import random

    # def add_gauss(points, var, max_width, max_height):
    width, height = 1280, 720
    N = 512
    points = np.zeros((N, 2))
    for i in range(N):
        points[i][0] = random.randint(0, width - 1)
        points[i][1] = random.randint(0, height - 1)

    var = 5.0
    noise_points = add_gauss(points, var, width, height)
    dif = noise_points - points

    f = plt.figure()
    plt.plot(dif[:, 0], dif[:, 1], '.')
    plt.show()


if __name__ == '__main__':
    # video_capture(
    # "/hdd/luke/hockey_data/USA 2-3 Canada - Men's Ice Hockey Gold Medal Match _ Vancouver 2010 Winter Olympics.mp4",
    #               "/hdd/luke/hockey_data/Olympic_2010/images/", 4072000, 25, 625)

    # video_capture(
    #     "/hdd/luke/hockey_data/Chicago Blackhawks VS Toronto Maple Leafs 15-01-2016  FULL.mp4",
    #     "/hdd/luke/hockey_data/Chicago_Toronto/images/", 1112500, 30, 1800)

    # video_capture(
    #     "/hdd/luke/hockey_data/Ice Hockey - Sweden 0 - 3 Canada - Men's Full Gold M"
    #     "edal Match _ Sochi 2014 Winter Olympics.mp4",
    #     "/hdd/luke/hockey_data/Olympic_2014/images/", 326000, 25, 800)

    # gt = load_camera_pose("../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat")
    # test = load_camera_pose("C:/graduate_design/experiment_result/new/soccer/all_rf.mat", True)
    #
    # gt = load_camera_pose("C:/graduate_design/experiment_result/new/synthesized/homography_keyframe_based/gt2.mat",
    #                       True)
    #
    # test = load_camera_pose(
    #     "C:/graduate_design/experiment_result/new/synthesized/homography_keyframe_based/outliers/50-keyframe.mat", True)
    gt = load_camera_pose("C:/graduate_design/dataset/synthesized/synthesize_ground_truth.mat",
                          True)
    test = load_camera_pose(
        "C:/graduate_design/experiment_result/baseline2/synthesized/ptz_ekf_tracking.mat", True)
    error = compute_error_data(test, gt)

    print(error)

    # ut_add_gaussian()
