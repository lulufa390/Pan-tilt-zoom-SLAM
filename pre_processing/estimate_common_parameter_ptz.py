
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

folder = '/Users/jimmy/Desktop/per_30_frames/'

import os
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
frame_numbers = get_immediate_subdirectories(folder)
frame_numbers.sort()

#frame_numbers = frame_numbers[0:20]
print(frame_numbers)
N = len(frame_numbers)

image_folder = '/Users/jimmy/Code/ptz_slam/dataset/hockey/UBC_2017/images/'
image_names = [image_folder + v + '.jpg' for v in frame_numbers]


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
print('common_center :{}'.format(common_center.shape))
print('common rotation: {}'.format(common_rotation.shape))


import scipy.io as sio

locations = init_cameras[:, 6:9]
location_ptzs = np.hstack((locations, opt_ptzs))

cameras = init_cameras
ptzs = opt_ptzs
cc = common_center
base_rotation = common_rotation
image_name = np.zeros((N), dtype=np.object)
for i in range(N):
    image_name[i] = '{}.jpg'.format(frame_numbers[i])


sio.savemat('UBC_hockey_ground_truth.mat',
            {'camera':cameras,
             'ptz':ptzs,
             'cc':cc,
             'base_rotation':base_rotation,
              'image_name':image_name})





"""
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
"""



