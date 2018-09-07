# general image processing functions

import cv2 as cv
import numpy as np


def detect_sift_corner(gray_img, nfeatures = 50):
    """
    :param gray_img:
    :param nfeatures:
    :return:          N x 2 matrix, sift keypoint location in the image
    """

    sift = cv.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    kp = sift.detect(gray_img, None)

    sift_pts = np.zeros((len(kp), 2))
    for i in range(len(kp)):
        sift_pts[i][0] = kp[i].pt[0]
        sift_pts[i][1] = kp[i].pt[1]

    return sift_pts

def detect_sift(im, nfeatures):
    """
    :param im:
    :param nfeatures:
    :return: two lists of key_point (2 dimension), and descriptor (128 dimension)
    """
    sift = cv.xfeatures2d.SIFT_create(nfeatures=nfeatures)
    key_point, descriptor = sift.detectAndCompute(im, None)

    # SIFT may detect more keypoint than set
    while len(key_point) > nfeatures:
        key_point.pop()
        descriptor.pop()

    return key_point, descriptor


def detect_harris_corner_grid(gray_img, row, column):
    """
    :param gray_img:
    :param row:  grid row number
    :param column:  grid column number
    :return:  a list of 1 x 2 matrix
    """
    mask = np.zeros_like(gray_img, dtype=np.uint8)
    h, w = gray_img.shape[0], gray_img.shape[1]
    grid_height = h // row
    grid_width = w // column

    all_harris = []

    for i in range(row):
        for j in range(column):

            grid_y1 = i * grid_height
            grid_x1 = j * grid_width

            if i == row - 1:
                grid_y2 = gray_img.shape[0]
            else:
                grid_y2 = i * grid_height + grid_height

            if j == column - 1:
                grid_x2 = gray_img.shape[1]
            else:
                grid_x2 = j * grid_width + grid_width

            mask[grid_y1:grid_y2, grid_x1:grid_x2] = 1
            grid_harris = cv.goodFeaturesToTrack(gray_img, maxCorners=5,
                                                 qualityLevel=0.2, minDistance=10, mask=mask.astype(np.uint8))
            all_harris.extend(grid_harris)
            mask[grid_y1:grid_y2, grid_x1:grid_x2] = 0  # reset mask
    return all_harris


def optical_flow_matching(img, next_img, points, ssd_threshold = 20):
    """
    :param img:    current image
    :param next_img: next image
    :param points: points on the current image
    :param ssd_threshold: optical flow parameters
    :return: matched index in the points, points in the next image. two lists
    """
    points = points.reshape((-1, 1, 2))  # 2D matrix to 3D matrix
    next_points, status, err = cv.calcOpticalFlowPyrLK(
        img, next_img, points.astype(np.float32), None, winSize=(31, 31))

    h, w = img.shape[0], img.shape[1]
    matched_index = []

    for i in range(len(next_points)):
        x, y = next_points[i][0][0], next_points[i][0][1]
        if err[i] < ssd_threshold and x >= 0 and x < w and y >= 0 and y < h:
            matched_index.append(i)

    next_points = [next_points[i][0] for i in matched_index]

    return matched_index, next_points



if __name__ == "__main__":
    im = cv.imread('../two_point_calib_dataset/highlights/seq1/0419.jpg', 0)
    print('image shape:', im.shape)

    # unit test
    pts = detect_sift_corner(im, 50)
    print(pts.shape)

    kp, des = detect_sift(im, 50)
    print(len(kp))
    print(len(des))
    print(des[0].shape)

    corners = detect_harris_corner_grid(im, 5, 5)
    print(len(corners))
    print(corners[0].shape)

    im1 = cv.imread('../two_point_calib_dataset/highlights/seq1/0419.jpg', 0)
    im2 = cv.imread('../two_point_calib_dataset/highlights/seq1/0422.jpg', 0)

    pts1 = detect_sift_corner(im1, 50)
    matched_index, next_points = optical_flow_matching(im1, im2, pts1, 20)

    print(len(matched_index), len(next_points))


    cv.imshow('image', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

