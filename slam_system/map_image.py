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


def get_median_ptz(ptz_list):
    """
    :param ptz_list: pan-tilt-zoom angles of length N, list of arrays with shape = 3.
    :return: a array of [pan, tilt, zoom]
    """
    pan_list = [angles[0] for angles in ptz_list]
    tilt_list = [angles[1] for angles in ptz_list]
    zoom_list = [angles[2] for angles in ptz_list]

    median_pan = np.median(pan_list)
    median_tilt = np.median(tilt_list)
    median_zoom = np.median(zoom_list)
    return np.array([median_pan, median_tilt, median_zoom])


def blending_with_median(image_list, mask_list):
    """
    Blending the images to a panorama with median color value at each pixel.
    :param image_list: a list of image to be combined.
    :param mask_list: a list of mask to annotate each pixel is inside a image or not.
    :return: blending result image.
    """
    assert len(image_list) == len(mask_list)
    for i in range(len(image_list)):
        assert image_list[i].shape == mask_list[i].shape

    blending_result = np.zeros(image_list[0].shape, np.uint8)

    for x in range(blending_result.shape[1]):
        for y in range(blending_result.shape[0]):
            for z in range(blending_result.shape[2]):
                color_value_list = []

                for i in range(len(mask_list)):
                    if mask_list[i][y, x, z] == 1:
                        color_value = image_list[i][y, x, z]
                        color_value_list.append(color_value)

                if len(color_value_list) > 0:
                    blending_result[y, x, z] = np.median(color_value_list)

                print(x, y, z)

    return blending_result


def blending_with_avg(image_list, mask_list):
    """
    Blending the images to a panorama with average color value at each pixel.
    :param image_list: a list of image to be combined.
    :param mask_list: a list of mask to annotate each pixel is inside a image or not.
    :return: blending result image.
    """
    assert len(image_list) == len(mask_list)
    for i in range(len(image_list)):
        assert image_list[i].shape == mask_list[i].shape

    sum_img = np.zeros(image_list[0].shape, np.uint16)
    total_mask = np.zeros(mask_list[0].shape, np.float)

    for i in range(len(image_list)):
        sum_img += image_list[i]
        total_mask += mask_list[i]

    total_mask[total_mask == 0] = 1

    panorama = (sum_img / total_mask).astype(np.uint8)

    return panorama


def generate_panoramic_image(standard_camera, img_list, ptz_list):
    """
    Generate panoramic image with a list of images and camera pose.
    :param standard_camera: a instance of PTZCamera, including shared parameters.
    :param img_list: image list of length N. They should be in the same shape and channels.
    :param ptz_list: Corresponding pan-tilt-zoom angles of length N.
    :return: a panoramic image.
    """

    assert len(img_list) == len(ptz_list)

    for image in img_list:
        assert len(image.shape) == 3

    vertical_border = 100
    horizontal_border = 500

    # the pan tilt zoom angles that all images project to
    # here it is set to be the median of all pans, tilts, zooms
    standard_ptz = get_median_ptz(ptz_list)

    # mask = 0 if it's in the border, else mask = 1
    mask = np.ones(img_list[0].shape, np.uint8)

    # wrap each image in enlarged_img_list with homography matrix
    dst_img_list = []

    # also wrap the mask
    wrap_mask_list = []

    for i, img in enumerate(img_list):
        # homography matrix
        matrix = get_wrap_matrix(standard_camera, ptz_list[i], standard_ptz)

        # transformation to right-down, to avoid being wrapped to axes' negative side
        trans_matrix = np.identity(3)
        trans_matrix[0, 2] = horizontal_border
        trans_matrix[1, 2] = vertical_border
        matrix = np.dot(trans_matrix, matrix)

        # wrapped images shape, larger than origin images
        dst_shape = (mask.shape[1] + horizontal_border * 2, mask.shape[0] + vertical_border * 2)

        # wrap origin image and mask
        dst = cv.warpPerspective(img, matrix, dst_shape)
        wrap_mask = cv.warpPerspective(mask, matrix, dst_shape)

        dst_img_list.append(dst)
        wrap_mask_list.append(wrap_mask)

    panorama = blending_with_avg(dst_img_list, wrap_mask_list)

    return panorama


def ut_basketball_map():
    seq = sio.loadmat("../../dataset/basketball/basketball_anno.mat")

    annotation = seq["annotation"]
    meta = seq["meta"]

    # the sequence to generate panorama
    img_sequence = [0, 600, 650, 700, 800, 900]

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


if __name__ == "__main__":
    ut_basketball_map()
