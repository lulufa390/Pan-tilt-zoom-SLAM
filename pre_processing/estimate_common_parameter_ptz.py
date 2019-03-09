
import scipy.io as sio
import numpy as np

import sys
sys.path.append('/Users/jimmy/Code/opencv_util/python_package')
import ntpath

# prepare calibration data
model = sio.loadmat('../gui/resource/ice_hockey_model.mat')
points = model['points']
model_pts = np.zeros((points.shape[0], 3))
model_pts[:, 0:2] = points
model_pts[:,2] = 0.0
rows, cols = model_pts.shape

folder = '/Users/jimmy/Desktop/sample5/'
frame_numbers = ['00048656', '00048818', '00049278', '00049916', '00050125']
N = len(frame_numbers)

image_names = [folder + v + '.jpg' for v in frame_numbers]

init_cameras = np.zeros((N, 9))

for i in range(N):
    file_name = folder + frame_numbers[i] + '/camera.txt'

    data = np.loadtxt(file_name, delimiter='\t', skiprows=2)
    init_cameras[i, :] = data


camera_num = N

init_commont_rotation = np.zeros((3, 3))

# main camera in the center
init_commont_rotation[0][0] = 1.0
init_commont_rotation[1][2] = -1.0
init_commont_rotation[2][1] = 1.0

import cv2 as cv
rod = np.zeros((3, 1))
cv.Rodrigues(init_commont_rotation, rod)



opt_cameras = np.zeros((N, 9))
opt_ptzs = np.zeros((N, 3))
common_center = np.zeros((3, 1))
common_rotation = np.zeros((3, 1))


from cvx_opt import optimize_ptz_cameras
opt_cameras, opt_ptzs, common_center, common_rotation = optimize_ptz_cameras(model_pts, init_cameras, rod)
print('opt_cameras :{}'.format(opt_cameras.shape))
print('opt_ptzs :{}'.format(opt_ptzs.shape))
print('common_center :{}'.format(common_center))
print('common rotation: {}'.format(common_rotation))



import scipy.io as sio


locations = init_cameras[:, 6:9]
location_ptzs = np.hstack((locations, opt_ptzs))

import sys
sys.path.append('../slam_system')
from visualize import project_model, broadcast_ptz_camera_project_model

model = sio.loadmat('../gui/resource/ice_hockey_model.mat')
points = model['points']
line_segment = model['line_segment_index']

lambda_dim = 12
shared_parameters = np.zeros((6 + lambda_dim, 1))
shared_parameters[0:3] = common_center
shared_parameters[3:6] = common_rotation

camera_param = np.zeros((6 + lambda_dim + 2 + 3, 1))
camera_param[0:6 + lambda_dim] = shared_parameters


for i in range(N):
    file_name = folder + frame_numbers[i] + '/camera.txt'
    im_name = image_names[i]

    im = cv.imread(im_name)

    init_cam = init_cameras[i, :]
    opt_cam = opt_cameras[i, :]
    im1 = project_model(init_cam, points, line_segment, im)

    pp = opt_cameras[i, 0:2].reshape((2, 1))
    ptz = opt_ptzs[i, :].reshape((3, 1))
    im2 = broadcast_ptz_camera_project_model(shared_parameters, pp, ptz, points, line_segment, im)

    cv.imshow('init camera', im1)
    cv.imshow('refined camera', im2)
    cv.waitKey()




