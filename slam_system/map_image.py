"""
This file is to generate a panoramic image as the map.

Created by Luke, 2019.1.8
"""

import cv2 as cv
import numpy as np
import scipy.io as sio
import numpy.linalg as lg

from ptz_camera import PTZCamera


def get_wrap_matrix(camera, src_ptz, target_ptz):
    """
    This function gets the homography matrix with two images' camera pose.
    The shared camera parameters are in 'camera'.
    :param camera: instance of <class 'PTZCamera'>
    :param src_ptz: array, shape = (3), pan-tilt-zoom angles for source image
    :param target_ptz: array, shape = (3), pan-tilt-zoom angles for target image
    :return: array, shape = (3, 3), homography matrix for cv2.wrapPerspective
    """

    camera.set_ptz(src_ptz)
    src_k = camera.compute_camera_matrix()
    src_rotation = camera.compute_rotation_matrix()

    camera.set_ptz(target_ptz)
    target_k = camera.compute_camera_matrix()
    target_rotation = camera.compute_rotation_matrix()

    # p1to2 is the homography matrix from img
    p1to2 = np.dot(target_k, np.dot(target_rotation, np.dot(lg.inv(src_rotation), lg.inv(src_k))))

    return p1to2


def enlarge_image(img, vertical, horizontal):
    """
    This function enlarges a image with black borders.
    :param img: source image to be enlarged.
    :param vertical: number of pixels to be enlarged vertically (increase of height is 2  * vertical).
    :param horizontal: number of pixels to be enlarged horizontally (increase of width is 2  * horizontal).
    :return: enlarged image with black border.
    """
    height = img.shape[0] + 2 * vertical
    width = img.shape[1] + 2 * horizontal

    if len(img.shape) == 3:
        enlarged_image = np.zeros((height, width, img.shape[2]), np.uint8)
    else:
        enlarged_image = np.zeros((height, width), np.uint8)

    enlarged_image[vertical:vertical + img.shape[0], horizontal:horizontal + img.shape[1]] = img

    return enlarged_image


def generate_panoramic_image(standard_camera, img_list, ptz_list):
    """
    Generate panoramic image with a list of images and camera pose.
    :param standard_camera: a instance of PTZCamera, including shared parameters.
    :param img_list: image list of length N. They should be in the same shape and channels.
    :param ptz_list: Corresponding pan-tilt-zoom angles of length N.
    :return: a panoramic image.
    """

    assert len(img_list) == len(ptz_list)

    # the pan tilt zoom angles that all images project to
    # here it is set to be the average of all pans, tilts, zooms
    standard_ptz = sum(ptz_list) / len(ptz_list)
    # standard_ptz = ptz_list[2]

    print(ptz_list)
    print(standard_ptz)

    border_vertical = 100
    border_horizontal = 500

    # enlarged image list to obligate enough space for homography transformation
    enlarged_img_list = []
    for img in img_list:
        enlarged_img = enlarge_image(img, border_vertical, border_horizontal)
        enlarged_img_list.append(enlarged_img)

    # mask = 0 if it's in the border, else mask = 1
    mask = np.ones(img_list[0].shape, np.uint8)
    enlarged_mask = enlarge_image(mask, border_vertical, border_horizontal)

    # wrap each image in enlarged_img_list with homography matrix
    dst_img_list = []

    # also wrap the mask
    wrap_mask_list = []

    for i, img in enumerate(enlarged_img_list):
        # homography matrix
        matrix = get_wrap_matrix(standard_camera, ptz_list[i], standard_ptz)

        # print(matrix)

        # wrap origin image and mask
        dst = cv.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

        dst2 = cv.warpPerspective(img, matrix, (img.shape[1] + 500, img.shape[0] + 200))
        cv.imshow("dst2", dst2)

        mask = cv.warpPerspective(enlarged_mask, matrix, (enlarged_mask.shape[1], enlarged_mask.shape[0]))

        dst_img_list.append(dst)
        wrap_mask_list.append(mask)

    # blending strategy: the intersection area is the average of all origin images.
    # so maintain a total_mask here as a count for number of images in the intersection area.
    # todo: may have better blending strategy (considering the distance).

    sum_img = np.zeros(enlarged_img_list[0].shape, np.uint16)
    total_mask = np.zeros(enlarged_img_list[0].shape, np.float)

    for i in range(len(dst_img_list)):
        # for wrap_mask in wrap_mask_list:
        #     panoramic_img += np.uint8(dst_img_list[i] * (wrap_mask / total_mask))

        sum_img += dst_img_list[i]
        total_mask += wrap_mask_list[i]

    total_mask[total_mask == 0] = 1

    panorama = (sum_img / total_mask).astype(np.uint8)
    return panorama


if __name__ == "__main__":
    seq = sio.loadmat("../../dataset/basketball/basketball_anno.mat")

    annotation = seq["annotation"]
    meta = seq["meta"]

    # the sequence to generate panorama
    img_sequence = [0,100, 600, 650, 670, 700,720,750,780,900]

    # shared parameters for ptz camera
    camera = PTZCamera(annotation[0][700]['camera'][0][0:2], meta[0][0]["cc"][0], meta[0][0]["base_rotation"][0])

    # get image list
    images = []
    for i in img_sequence:
        img = cv.imread("../../dataset/basketball/images/" + annotation[0][i]['image_name'][0], 1)
        images.append(img)

    # get camera pose list (corresponding to image list)
    ptz_list = []
    for i in img_sequence:
        ptz_list.append(annotation[0][i]['ptz'][0])

    panorama = generate_panoramic_image(camera, images, ptz_list)
    cv.imshow("test", panorama)

    cv.imwrite("../../map/panorama.jpg", panorama)
    cv.waitKey(0)
