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

def project_model(camera, model_points, model_line_segment, rgb_image):
    """
    project a 2D field model to the image space
    :param camera: 9 camera parameters, (u, v, focal_length, Rx, Rx, Rz, Cx, Cy, Cz)
                    (Rx, Rx, Rz): is Rodrigues angle, (Cx, Cy, Cz) unit in meter
    :param model_points: N x 2 matrix, model world coordinate in meter
    :param model_line_segment: ine segment index, start from 0
    :param rgb_image: an RGB image
    :return:
    % example:
    % camera = [640.000000	 360.000000	 2986.943295	 1.367497	 -1.082443	 0.980122	 -16.431519	 14.086604	 5.580546];
    % I = imread('00003600.jpg');
    % load('soccer_field_model.mat');
    % model_points = points;
    % model_line_segment = line_segment_index + 1;
    % project_model(camera, model_points, model_line_segment, I);
    """
    assert camera.shape[0] == 9
    assert model_points.shape[1] == 2
    assert model_line_segment.shape[1] == 2
    u, v, f = camera[0:3]
    K = np.eye(3, 3)
    K[0][0] = f
    K[0][2] = u
    K[1][1] = f
    K[1][2] = v
    rod = camera[3:6]
    rotation = np.zeros((3, 3))
    cv.Rodrigues(rod, rotation)
    cc = camera[6:9]
    #print('rotation', rotation)
    N = model_points.shape[0]
    image_points = np.zeros((N, 2))
    for i in range(N):
        p = np.array([model_points[i][0], model_points[i][1], 0])
        p = np.dot(K, np.dot(rotation, p - cc))
        if p[2] != 0.0:
            image_points[i][0] = p[0] / p[2]
            image_points[i][1] = p[1] / p[2]
    import  copy
    vis_image = copy.deepcopy(rgb_image)
    for i in range(len(model_line_segment)):
        begin = int(model_line_segment[i][0])
        end = int(model_line_segment[i][1])
        cv.line(vis_image, (int(image_points[begin][0]), int(image_points[begin][1])),
                (int(image_points[end][0]), int(image_points[end][1])), (0, 0, 255), 2)
    return vis_image

def broadcast_ptz_camera_project_model(camera, model_points, model_line_segment, rgb_image):
    """
    project a 2D field model to the image space
    :param camera: 17 parameters
    shared (camera center, rotation, lambda), principal point, pan-tilt-zoom,  12 + 2 + 3 = 17
    camera center in meters
    :param model_points: N x 2 matrix, model world coordinate in meter
    :param model_line_segment: ine segment index, start from 0
    :param rgb_image: an RGB image
    :return: rgb image with overlaid line model
    """
    from ctypes import cdll
    from ctypes import c_int
    from ctypes import c_void_p
    #@todo hardcode library
    lib = cdll.LoadLibrary('/Users/jimmy/Source/opencv_util/build/libcvx_opt_python.dylib')
    assert camera.shape[0] == 17
    assert model_points.shape[1] == 2
    assert model_line_segment.shape[1] == 2
    N = model_points.shape[0]
    image_points = np.zeros((N, 2))
    points = np.zeros((N, 3))
    points[:,:-1] = model_points

    lib.broadcastCameraProjection(c_void_p(camera.ctypes.data),
                                  c_void_p(points.ctypes.data),
                                  c_int(N),
                                  c_void_p(image_points.ctypes.data))
    print(image_points)
    import copy
    vis_image = copy.deepcopy(rgb_image)
    for i in range(len(model_line_segment)):
        begin = int(model_line_segment[i][0])
        end = int(model_line_segment[i][1])
        cv.line(vis_image, (int(image_points[begin][0]), int(image_points[begin][1])),
                (int(image_points[end][0]), int(image_points[end][1])), (0, 0, 255), 2)
    return vis_image


def ut_project_model():
    camera = np.array([640.000000, 360.000000, 2986.943295,
                       1.367497, -1.082443, 0.980122,
                       -16.431519, 14.086604, 5.580546])
    image = cv.imread('/Users/jimmy/Desktop/00003600.jpg')
    data = sio.loadmat('/Users/jimmy/Desktop/wwos_soccer_field_model.mat')

    model_points = data['points']
    model_line_segment = data['line_segment_index']
    image = project_model(camera, model_points, model_line_segment, image)
    cv.imshow('image', image)
    cv.waitKey(0)


    # I = imread('00003600.jpg');
    # load('soccer_field_model.mat');
    # model_points = points;
    # model_line_segment = line_segment_index + 1;
    # project_model(camera, model_points, model_line_segment, I);

def ut_broadcast_ptz_camera_project_model():
    pass



def ut_Visualize():
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


if __name__ == '__main__':
    ut_project_model()
