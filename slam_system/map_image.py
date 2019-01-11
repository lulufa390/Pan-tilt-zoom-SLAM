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
    :param src_ptz: array: shape = [3], pan tilt zoom for source
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


def enlarge_image(img, vertical, horizontal):
    """
    :param img:
    :param vertical:
    :param horizontal:
    :return:
    """
    height = img.shape[0] + 2*vertical
    width = img.shape[1] + 2*horizontal

    if len(img.shape) == 3:
        enlarged_image = np.zeros((height, width, img.shape[2]), np.uint8)
    else:
        enlarged_image = np.zeros((height, width), np.uint8)

    enlarged_image[vertical:vertical+img.shape[0], horizontal:horizontal+img.shape[1]] = img


    return enlarged_image


def generate_panoramic_image(standard_camera, img_list, ptz_list):
    """
    generate l panoramic image with a list of images and camera pose.
    :param standard_camera: a instance of PTZCamera. The camera of image in the middle.
    :param img_list: image list of length N. They should be in the same size.
    :param ptz_list: Corresponding pan-tilt-zoom list of length N.
    :return: a panoramic image.
    """

    assert len(img_list) == len(ptz_list)

    img_num = len(img_list)

    standard_ptz = standard_camera.get_ptz()

    height = img_list[0].shape[0]
    width = img_list[0].shape[1]

    border_vertical = 50
    border_horizontal = 200

    width_with_border = width + border_horizontal * 2
    height_with_border = height + border_vertical * 2

    mask_list = []
    enlarged_img_list = []
    for img in img_list:
        enlarged_img_list.append(enlarge_image(img, border_vertical, border_horizontal))
        mask_list.append(enlarge_image(np.zeros(img.shape, np.uint8), border_vertical, border_horizontal))

    dst_img_list = []
    wrap_mask_list = []
    for index, img in enumerate(enlarged_img_list):
        matrix = get_wrap_matrix(standard_camera, standard_ptz, ptz_list[index])

        dst = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
        mask = cv.warpPerspective(mask_list[index], matrix, (mask_list[index].shape[1], mask_list[index].shape[0]))

        dst_img_list.append(dst)
        wrap_mask_list(wrap_mask_list)

    panoramic_img = np.zeros(img_list[0].shape, np.uint8)
    total_mask = np.zeros(img_list[0].shape, np.uint8)
    for i in range(len(dst_img_list)):
        panoramic_img += dst_img_list[i]
        total_mask += wrap_mask_list[i]

    return panoramic_img / total_mask


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

    print(output.shape)

    dst = cv.warpPerspective(im1, matrix, (im1.shape[1]  , im1.shape[0] ))

    cv.imshow("dst", im1)

    cv.imshow("enlarged", enlarge_image(im1, 20, 40))

    # combine images
    # for i in range(im1.shape[0]):
    #     for j in range(im1.shape[1]):
    #         if dst[i, j, 0] < 10 and dst[i, j, 1] < 10 and dst[i, j, 2] < 10:
    #             output[i, j] = im2[i, j]
    #         else:
    #             output[i, j] = 0.5 * dst[i, j] + 0.5 * im2[i, j]
    #
    # cv.imshow("output", output)
    cv.waitKey(0)

    pass
