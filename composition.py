import scipy.io as sio
import cv2 as cv
import numpy as np
import numpy.linalg as lg

#load the information for one sequence

path = "./two_point_calib_dataset/highlights/seq4"
i1 = 10
i2 = 3

seq = sio.loadmat(path + "_anno.mat")

#image name, image center, f, rotation(3), base(3), ...
annotation = seq["annotation"]
meta = seq['meta']

#load the soccer model data
soccer_model = sio.loadmat("./two_point_calib_dataset/util/highlights_soccer_model.mat")

line_index = soccer_model['line_segment_index']
points = soccer_model['points']


img1 = cv.imread(path + "/" + annotation[0][i1]['image_name'][0] , 1)
img2 = cv.imread(path + "/" + annotation[0][i2]['image_name'][0] , 1)


camera1 = annotation[0][i1]['camera'][0]
camera2 = annotation[0][i2]['camera'][0]

K1_paras = camera1[0:3]
K1 = np.array([[ K1_paras[2],0, K1_paras[0]],[0, K1_paras[2], K1_paras[1]],[0,0,1]])

K2_paras = camera2[0:3]
K2 = np.array([[ K2_paras[2],0, K2_paras[0]],[0, K2_paras[2], K2_paras[1]],[0,0,1]])

rotation1 = np.zeros([3, 3])
cv.Rodrigues(camera1[3:6], rotation1)

rotation2 = np.zeros([3, 3])
cv.Rodrigues(camera2[3:6], rotation2)

# cv.imshow("img1", img1)
# cv.imshow("img2", img2)

p1to2 = np.dot(K2,  np.dot( rotation2 ,np.dot(lg.inv(rotation1), lg.inv(K1))))
p2to1 = np.dot(K1,  np.dot( rotation1 ,np.dot(lg.inv(rotation2), lg.inv(K2))))

output = np.zeros(img1.shape, np.uint8)

dst = cv.warpPerspective(img1,p1to2,(img1.shape[1], img1.shape[0]))
# cv.imshow("fff",dst//2 + img2//2)
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if dst[i,j,0] < 10 and dst[i,j,1] < 10 and dst[i,j,2] < 10:
            output[i,j] = img2[i,j]
        else:
            output[i,j] = 0.5*dst[i,j] + 0.5*img2[i,j]


# dst = cv.warpPerspective(img2,p2to1,(img2.shape[1], img2.shape[0]))
# # cv.imshow("fff",  dst//2 + img1//2)
# for i in range(img1.shape[0]):
#     for j in range(img1.shape[1]):
#         if dst[i,j,0] < 10 and dst[i,j,1] < 10 and dst[i,j,2] < 10:
#             output[i,j] = img1[i,j]
#         else:
#             output[i,j] = 0.5*dst[i,j] + 0.5*img1[i,j]

cv.imshow("output",output)
cv.waitKey(0)