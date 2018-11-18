
import scipy.io as sio
import numpy as np
import ctypes
from ctypes import cdll
from ctypes import c_int
lib = cdll.LoadLibrary('/Users/jimmy/Source/opencv_util/build/libcvx_opt_python.dylib')


# prepare calibration data
#model = sio.loadmat('./data/ice_hockey_edge_point_6_feet.mat')
#edge_points = model['edge_points']
#model = sio.loadmat('./data/worldcup_2014_model.mat')
model = sio.loadmat('./data/ice_hockey_model.mat')
points = model['points']
model_pts = np.zeros((points.shape[0], 3))
model_pts[:, 0:2] = points
model_pts[:,2] = 0.0
rows, cols = model_pts.shape


import glob
#files = glob.glob('./data/bra_mex/*.txt')
#files = glob.glob('/Users/jimmy/Desktop/Chicago_Totonto_reference_frame/annotation/*.txt')
files = glob.glob('./data/annotation/*.txt')
N = len(files)
# initial camera data

init_cameras = np.zeros((N, 9))
for i in range(N):
    file_name = files[i]
    data = np.loadtxt(file_name, delimiter='\t', skiprows=2)
    init_cameras[i, :] = data

#sio.savemat('init_cameras.mat', {'init_camera':init_cameras})

camera_num, camera_param_len = N, 9

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
shared_parameters = np.zeros((12, 1))


lib.estimateCommonCameraCenterAndRotationAndDisplacment(ctypes.c_void_p(model_pts.ctypes.data),
                                                        c_int(rows),
                                                        ctypes.c_void_p(init_cameras.ctypes.data),
                                                        c_int(camera_num),
                                                        ctypes.c_void_p(rod.ctypes.data),
                                                        ctypes.c_void_p(opt_cameras.ctypes.data),
                                                        ctypes.c_void_p(opt_ptzs.ctypes.data),
                                                        ctypes.c_void_p(shared_parameters.ctypes.data),)

#print(opt_cameras[0,:])
#print(shared_parameters)

locations = init_cameras[:, 6:9]
location_ptzs = np.hstack((locations, opt_ptzs))

#sio.savemat('location_ptz.mat', {'location_ptz': location_ptzs})

import sys
sys.path.append('../slam_system')
from visualize import project_model, broadcast_ptz_camera_project_model

#model = sio.loadmat('./data/ice_hockey_model.mat')
points = model['points']
line_segment = model['line_segment_index']

camera_param = np.zeros((17, 1))
camera_param[0:12] = shared_parameters

for i in range(len(files)):
    file = open(files[i], 'r')
    im_name = file.readline()
    im_name = im_name[:-1]
    #im_name = im_name.replace('images', 'image_data', 1)

    im = cv.imread(im_name)

    init_cam = init_cameras[i, :]
    opt_cam = opt_cameras[i, :]
    im1 = project_model(init_cam, points, line_segment, im)
    camera_param[12:14] = opt_cameras[i, 0:2].reshape((2,1))
    camera_param[14:17] = opt_ptzs[i,:].reshape((3, 1))
    im2 = broadcast_ptz_camera_project_model(camera_param, points, line_segment, im)

    cv.imshow('init camera', im1)
    cv.imshow('refined camera', im2)
    cv.waitKey()












