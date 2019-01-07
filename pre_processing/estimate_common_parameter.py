
import scipy.io as sio
import numpy as np

import sys
sys.path.append('/Users/jimmy/Source/opencv_util/python_package')
import ntpath

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

image_dir = './data/olympic_2010_reference_frame/image/'
import glob
#files = glob.glob('./data/bra_mex/*.txt')
#files = glob.glob('/Users/jimmy/Desktop/Chicago_Totonto_reference_frame/annotation/*.txt')
#files = sorted(glob.glob('./data/olympic_2010_reference_frame/annotation/*.txt'))
files = glob.glob('./data/olympic_2010_reference_frame/annotation/*.txt')
N = len(files)
# initial camera data

init_cameras = np.zeros((N, 9))
images = []
for i in range(N):
    file_name = files[i]

    data = np.loadtxt(file_name, delimiter='\t', skiprows=2)
    init_cameras[i, :] = data

    file = open(files[i], 'r')
    im_name = file.readline()
    im_name = im_name[:-1]
    im_name = ntpath.basename(im_name)
    images.append(im_name)


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


lambda_dim = 12
opt_cameras = np.zeros((N, 9))
opt_ptzs = np.zeros((N, 3))
shared_parameters = np.zeros((6 + lambda_dim, 1))


from cvx_opt import optimize_broadcast_cameras
opt_cameras, opt_ptzs, shared_parameters = optimize_broadcast_cameras(model_pts, init_cameras, rod, lambda_dim)
print('opt_cameras :{}'.format(opt_cameras.shape))
print('opt_ptzs :{}'.format(opt_ptzs.shape))
print('shared_parameters :{}'.format(shared_parameters.shape))


"""
# print displacement
displacement = np.zeros((N, 4))
for i in range(opt_ptzs.shape[0]):
    wt = shared_parameters[6:12]
    f = opt_ptzs[i][2]
    dx = wt[0] + wt[3] * f
    dy = wt[1] + wt[4] * f
    dz = wt[2] + wt[5] * f
    displacement[i] = dx, dy, dz, opt_ptzs[i][1]
    #print("dx dy dz: {} {} {}".format(dx, dy, dz))
"""

import scipy.io as sio
#sio.savemat('olympic_2010_reference_frame.mat', {'opt_cameras':opt_cameras,
#                                                 'opt_ptzs':opt_ptzs,
#                                                 'shared_parameters':shared_parameters,
#                                                  'images': images})

#sio.savemat('displacement.mat', {'displacement':displacement})

#sio.savemat('init_camera_26.mat', {'init_camera':init_cameras})

#print('shared parameters : {}'.format(shared_parameters))



locations = init_cameras[:, 6:9]
location_ptzs = np.hstack((locations, opt_ptzs))

#sio.savemat('location_ptz.mat', {'location_ptz': location_ptzs})

import sys
sys.path.append('../slam_system')
from visualize import project_model, broadcast_ptz_camera_project_model

model = sio.loadmat('./data/ice_hockey_model.mat')
points = model['points']
line_segment = model['line_segment_index']

camera_param = np.zeros((6 + lambda_dim + 2 + 3, 1))
camera_param[0:6 + lambda_dim] = shared_parameters


import ntpath
for i in range(len(files)):
    file = open(files[i], 'r')
    im_name = file.readline()
    im_name = im_name[:-1]
    im_name = ntpath.basename(im_name)
    im_name = image_dir + im_name
    #print(im_name)


    im = cv.imread(im_name)

    init_cam = init_cameras[i, :]
    opt_cam = opt_cameras[i, :]
    im1 = project_model(init_cam, points, line_segment, im)

    pp = opt_cameras[i, 0:2].reshape((2,1))
    ptz = opt_ptzs[i,:].reshape((3, 1))
    im2 = broadcast_ptz_camera_project_model(shared_parameters, pp, ptz, points, line_segment, im)

    cv.imshow('init camera', im1)
    cv.imshow('refined camera', im2)
    cv.waitKey()
















