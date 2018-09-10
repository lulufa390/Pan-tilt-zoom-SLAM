"""
Bundle Adjustment function
"""

import scipy.io as sio
import cv2 as cv
import numpy as np
from key_frame import KeyFrame
from image_process import build_matching_graph
from sequence_manager import SequenceManager


def bundle_adjustment(keyframes):
    """
    optimize global rays, camera pose in each key frames
    :param keyframes:
    :return:  optimized keyframes
    """
    pass


def ut_test_build_adjustment():

    input = SequenceManager("/Users/jimmy/Desktop/ptz_slam_dataset/basketball/basketball_anno.mat",
                            "/Users/jimmy/Desktop/ptz_slam_dataset/basketball/images",
                            "/Users/jimmy/PycharmProjects/ptz_slam/Camera-Calibration/basketball/objects_basketball.mat")

    cc = input.get_camera_center()
    base_rotation = input.get_base_rotation()
    u = 1280/2
    v = 720/2

    image_index = [0, 660, 700, 740, 800]
    images = []

    key_frames = []
    # initialize key frames
    for i in range(len(image_index)):
        im = input.get_basketball_image(image_index[i])
        ptz = input.get_ptz(image_index[i])
        key_frame = KeyFrame(im, i, cc, base_rotation, u, v, ptz[0], ptz[1], ptz[2])
        images.append(im)

        key_frames.append(key_frame)



    keypoints, descriptors, landmark_index = build_matching_graph(images, True)


if __name__ == '__main__':
    ut_test_build_adjustment()
