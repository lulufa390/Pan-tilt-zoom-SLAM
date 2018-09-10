"""
Bundle Adjustment function
"""

import scipy.io as sio
import cv2 as cv
import numpy as np
import math
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

    image_index = [0, 660, 710, 730, 800]

    N = len(image_index)
    key_frames = []
    # initialize key frames
    for i in range(len(image_index)):
        im = input.get_basketball_image(image_index[i])
        ptz = input.get_ptz(image_index[i])
        key_frame = KeyFrame(im, i, cc, base_rotation, u, v, ptz[0], ptz[1], ptz[2])
        key_frames.append(key_frame)
        #print(ptz)

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

    #print(image_match_mask)
    keypoints, descriptors, landmark_index = build_matching_graph(images, image_match_mask, True)


if __name__ == '__main__':
    ut_test_build_adjustment()
