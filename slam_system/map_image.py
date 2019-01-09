"""
This is the file to generate a panoramic image as map.

Created by Luke, 2019.1.8
"""

import cv2 as cv
import numpy as np
import scipy.io as sio
import numpy.linalg as lg
from ptz_camera import PTZCamera


def get_wrap_matrix(camera, src_ptz, target_ptz):
    """
    This function get the wrapPerspective matrix from source camera pose to target camera pose
    :param camera: instance of PTZCamera
    :param src_ptz: array: shape = [3], pan-
    :param target_ptz: array, shape = [3], pan tilt zoom for target
    :return: [3, 3] matrix for wrapPerspective
    """

    camera.set_ptz(src_ptz)

    src_k = camera.compute_camera_matrix()
    src_rotation = camera.compute_rotation_matrix()

    camera.set_ptz(target_ptz)
    target_k = camera.compute_camera_matrix()
    target_rotation = camera.compute_rotation_matrix()

    # projection matrix
    p1to2 = np.dot(target_k, np.dot(target_rotation, np.dot(lg.inv(src_rotation), lg.inv(src_k))))

    return p1to2


if __name__ == "__main__":
    seq = sio.loadmat("../../dataset/basketball/basketball_anno.mat")
    # image name, image center, f, rotation(3), base(3), ...
    annotation = seq["annotation"]
    meta = seq["meta"]

    i1 = 0
    i2 = 700

    im1 = cv.imread("../../dataset/basketball/images/" + annotation[0][i1]['image_name'][0], 1)
    im2 = cv.imread("../../dataset/basketball/images/" + annotation[0][i2]['image_name'][0], 1)

    camera = PTZCamera(annotation[0][i1]['camera'][0][0:2], meta[0][0]["cc"][0], meta[0][0]["base_rotation"][0])

    src_ptz = annotation[0][i1]['ptz'][0]
    target_ptz = annotation[0][i2]['ptz'][0]

    matrix = get_wrap_matrix(camera, src_ptz, target_ptz)

    output = np.zeros(im1.shape, np.uint8)
    dst = cv.warpPerspective(im1, matrix, (im1.shape[1], im1.shape[0]))

    # combine images
    for i in range(im1.shape[0]):
        for j in range(im1.shape[1]):
            if dst[i, j, 0] < 10 and dst[i, j, 1] < 10 and dst[i, j, 2] < 10:
                output[i, j] = im2[i, j]
            else:
                output[i, j] = 0.5 * dst[i, j] + 0.5 * im2[i, j]

    cv.imshow("output", output)
    cv.waitKey(0)

    pass
