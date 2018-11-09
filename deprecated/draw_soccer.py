import scipy.io as sio
import cv2 as cv
import numpy as np

"""
Function for drawing the lines on soccer fields using:
1. the 3D model 
2. ground truth camera pose for some images 
"""


def visualize_line(path):
    # load the annotations for some of the images in one sequence
    seq = sio.loadmat(path + "_anno.mat")
    # image name, image center, f, rotation(3), base(3), ...
    annotation = seq["annotation"]
    # other fixed information for this sequence
    meta = seq['meta']

    # load the soccer model data
    soccer_model = sio.loadmat("./two_point_calib_dataset/util/highlights_soccer_model.mat")
    # points position
    points = soccer_model['points']
    # the indexes for points to draw lines
    line_index = soccer_model['line_segment_index']

    # for each annotated images, draw lines based on ground truth camera pose
    for i in range(annotation.size):

        # load image
        img = cv.imread(path + "/"
                        + annotation[0][i]['image_name'][0], 1)

        # get camera information
        camera = annotation[0][i]['camera'][0]

        # intrinsic matrix
        k_paras = camera[0:3]
        k = np.array([[k_paras[2], 0, k_paras[0]], [0, k_paras[2], k_paras[1]], [0, 0, 1]])

        # rotation matrix
        rotation = np.zeros([3, 3])
        cv.Rodrigues(camera[3:6], rotation)

        # base position
        c = np.array(camera[6:9])

        # points coordinates
        image_points = np.ndarray([len(points), 2])

        for j in range(len(points)):
            p = np.array([points[j][0], points[j][1], 0])
            p = np.dot(k, np.dot(rotation, p - c))
            image_points[j][0] = p[0] / p[2]
            image_points[j][1] = p[1] / p[2]

        # draw lines
        for j in range(len(line_index)):
            begin = line_index[j][0]
            end = line_index[j][1]

            cv.line(img, (int(image_points[begin][0]), int(image_points[begin][1])),
                    (int(image_points[end][0]), int(image_points[end][1])), (0, 0, 255), 5)

        cv.imshow("result", img)
        cv.waitKey(0)

    return


visualize_line("./two_point_calib_dataset/highlights/seq4")
