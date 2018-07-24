import scipy.io as sio
import cv2 as cv
import numpy as np

#load the information for one sequence
seq = sio.loadmat("./two_point_calib_dataset/highlights/seq1_anno.mat")


#image name, image center, f, rotation(3), base(3), ...
annotation = seq["annotation"]
meta = seq['meta']

#load the soccer model data
soccer_model = sio.loadmat("./two_point_calib_dataset/util/highlights_soccer_model.mat")

line_index = soccer_model['line_segment_index']
points = soccer_model['points']

print(len(line_index))
print(annotation.size)
for i in range(annotation.size):
    img = cv.imread("./two_point_calib_dataset/highlights/seq1/"
                    + annotation[0][i]['image_name'][0] , 1)
    print(i)
    camera = annotation[0][i]['camera'][0]
    # print(camera)


    K_paras = camera[0:3]
    K = np.array([[ K_paras[2],0, K_paras[0]],[0, K_paras[2], K_paras[1]],[0,0,1]])
    # print(K)

    # cv.rod
    rotation = np.zeros([3,3])
    cv.Rodrigu es(camera[3:6], rotation)

    #base c
    c = np.array(camera[6:9])
    print("c", c)

    image_points = np.ndarray([len(points), 2])

    for j in range(len(points)):
        p = np.array([points[j][0], points[j][1],0])
        p = np.dot(K , np.dot(rotation , p - c ))
        # print(p.shape)
        x = p[0] / p[2]
        y = p[1] / p[2]

        image_points[j][0] = x
        image_points[j][1] = y

        # print(j, p)


    # print(image_points)

    for j in range(len(line_index)):
        begin = line_index[j][0]
        end = line_index[j][1]

        cv.line(img, (int(image_points[begin][0]), int(image_points[begin][1])),
                (int(image_points[end][0]), int(image_points[end][1])), (0,0  ,255), 5)


    cv.imshow("test", img)
    cv.waitKey(0)
