import scipy.io as sio
import cv2 as cv
import numpy as np
import numpy.linalg as lg

"""
Test program to composite two images using their camera pose. 
img_loc1 = k1 * r1 * (pos - c) and img_loc2 = k2 * r2 * (pos- c)
"""


def composition(path, i1, i2):
    # load the information for one sequence
    seq = sio.loadmat(path + "_anno.mat")
    # image name, image center, f, rotation(3), base(3), ...
    annotation = seq["annotation"]

    # select two images
    img1 = cv.imread(path + "/" + annotation[0][i1]['image_name'][0], 1)
    img2 = cv.imread(path + "/" + annotation[0][i2]['image_name'][0], 1)

    # get two camera locations
    camera1 = annotation[0][i1]['camera'][0]
    camera2 = annotation[0][i2]['camera'][0]

    # intrinsic matrix
    k1_paras = camera1[0:3]
    K1 = np.array([[k1_paras[2], 0, k1_paras[0]], [0, k1_paras[2], k1_paras[1]], [0, 0, 1]])
    k2_paras = camera2[0:3]
    k2 = np.array([[k2_paras[2], 0, k2_paras[0]], [0, k2_paras[2], k2_paras[1]], [0, 0, 1]])

    # rotation matrix
    rotation1 = np.zeros([3, 3])
    cv.Rodrigues(camera1[3:6], rotation1)
    rotation2 = np.zeros([3, 3])
    cv.Rodrigues(camera2[3:6], rotation2)

    # projection matrix
    p1to2 = np.dot(k2, np.dot(rotation2, np.dot(lg.inv(rotation1), lg.inv(K1))))

    output = np.zeros(img1.shape, np.uint8)
    dst = cv.warpPerspective(img1, p1to2, (img1.shape[1], img1.shape[0]))

    # combine images
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if dst[i, j, 0] < 10 and dst[i, j, 1] < 10 and dst[i, j, 2] < 10:
                output[i, j] = img2[i, j]
            else:
                output[i, j] = 0.5 * dst[i, j] + 0.5 * img2[i, j]

    cv.imshow("output", output)
    cv.waitKey(0)


composition("./two_point_calib_dataset/highlights/seq4", 3, 10)
