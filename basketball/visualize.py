"""
Visualize class to draw red lines based on estimated camera pose.

Created by Luke, 2018.9
"""

import cv2 as cv
import scipy.io as sio
import numpy as np
from math import *
from sequence_manager import SequenceManager


class Visualize:
    def __init__(self, model_path, annotation_path, image_path):
        """
        :param model_path: court model path
        :param annotation_path: annotation file path
        :param image_path: image folder path
        """
        soccer_model = sio.loadmat(model_path)
        self.line_index = soccer_model['line_segment_index']
        self.points = soccer_model['points']

        self.sequence = SequenceManager(annotation_path, image_path)

    def draw_line(self, img, p, t, f):
        """
        :param img: color image of this frame
        :param p: camera pose pan
        :param t: camera pose tilt
        :param f: camera pose focal length
        """
        k = np.array([[f, 0, self.sequence.u], [0, f, self.sequence.v], [0, 0, 1]])
        pan = radians(p)
        tilt = radians(t)
        rotation = np.dot(np.array([[1, 0, 0],
                                    [0, cos(tilt), sin(tilt)],
                                    [0, -sin(tilt), cos(tilt)]]),
                          np.array([[cos(pan), 0, -sin(pan)],
                                    [0, 1, 0],
                                    [sin(pan), 0, cos(pan)]]))
        rotation = np.dot(rotation, self.sequence.base_rotation)
        image_points = np.ndarray([len(self.points), 2])
        for j in range(len(self.points)):
            p = np.array([self.points[j][0], self.points[j][1], 0])
            p = np.dot(k, np.dot(rotation, p - self.sequence.c))
            image_points[j][0] = p[0] / p[2]
            image_points[j][1] = p[1] / p[2]
        for j in range(len(self.line_index)):
            begin = int(self.line_index[j][0])
            end = int(self.line_index[j][1])
            cv.line(img, (int(image_points[begin][0]), int(image_points[begin][1])),
                    (int(image_points[end][0]), int(image_points[end][1])), (0, 0, 255), 10)


if __name__ == '__main__':
    visualize = Visualize("./basketball/basketball_model.mat",
                          "./basketball/basketball/basketball_anno.mat", "./basketball/basketball/images")

    # visualize = Visualize("./two_point_calib_dataset/util/highlights_soccer_model.mat",
    #                       "./two_point_calib_dataset/highlights/seq3_anno.mat", "./seq3_blur/")

    camera_pos = sio.loadmat("../result/basketball/DoG-50+SIFT/camera_pose.mat")
    predict_pan = camera_pos['predict_pan'].squeeze()
    predict_tilt = camera_pos['predict_tilt'].squeeze()
    predict_f = camera_pos['predict_f'].squeeze()

    for i in range(visualize.sequence.anno_size):
        # for i in range(333):
        img = visualize.sequence.get_image(i, 0)
        visualize.draw_line(img, predict_pan[i], predict_tilt[i], predict_f[i])

        cv.imwrite("/hdd/luke/visualize/basketball/" + str(i) + ".jpg", img)
        print(i)
        # cv.imshow("test", img)
        # cv.waitKey(0)
