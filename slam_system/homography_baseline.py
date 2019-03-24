"""
Baseline:
Homography based frame-to-frame matching

"""

import scipy.io as sio
import cv2 as cv
import copy

from sequence_manager import SequenceManager
from scene_map import Map
from key_frame import KeyFrame
from relocalization import relocalization_camera
from ptz_camera import PTZCamera
from image_process import *
from util import *
from visualize import *


class HomographyTracking:
    def __init__(self, first_frame, first_frame_matrix):
        self.current_frame = first_frame
        self.first_matrix = first_frame_matrix

        self.accumulate_matrix = [first_frame_matrix, ]
        self.each_homography = [None, ]

    def tracking(self, next_frame):
        kp1, des1 = detect_compute_sift(self.current_frame, 0)
        kp2, des2 = detect_compute_sift(next_frame, 0)

        self.current_frame = next_frame

        homography = compute_homography(kp1, des1, kp2, des2)

        self.each_homography.append(homography)
        self.accumulate_matrix.append(np.dot(homography, self.accumulate_matrix[-1]))


if __name__ == "__main__":
    sequence = SequenceManager("../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_330",
                               "../../dataset/soccer_dataset/seq3/seq3_ground_truth.mat",
                               "../../dataset/soccer_dataset/seq3/seq3_player_bounding_box.mat")

    line_index, points = load_model("../../dataset/soccer_dataset/highlights_soccer_model.mat")

    first_frame_ptz = (sequence.ground_truth_pan[0],
                       sequence.ground_truth_tilt[0],
                       sequence.ground_truth_f[0])

    first_camera = sequence.camera
    first_camera.set_ptz(first_frame_ptz)

    # 3*4 projection matrix for 1st frame
    first_frame_mat = first_camera.projection_matrix
    first_frame = sequence.get_image_gray(index=0, dataset_type=1)

    img = project_with_homography(first_frame_mat, points, line_index, first_frame)

    cv.imshow("image", img)
    cv.waitKey()

    tracking_obj = HomographyTracking(first_frame, first_frame_mat)

    for i in range(1, sequence.length):
        next_frame = sequence.get_image_gray(index=i, dataset_type=1)
        tracking_obj.tracking(next_frame)

        img = project_with_homography(tracking_obj.accumulate_matrix[-1], points, line_index, next_frame)

        cv.imshow("image", img)
        cv.waitKey(100)
