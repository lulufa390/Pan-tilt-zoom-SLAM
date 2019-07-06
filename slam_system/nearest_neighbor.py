"""
Created by Jikai Lu, 2019.7

A nearest neighbor search based relocalization solution.
"""

import numpy as np
import pyflann

from scene_map import Map
from ptz_camera import PTZCamera


class NNBasedMap(Map):
    def __int__(self):
        # [N, 128] descriptor array
        self.global_des = np.ndarray([0, 128])

        # init flann
        self.flann = pyflann.FLANN()
        self.params = None

    def build_kdtree(self):
        self.params = self.flann.build_index(self.global_des, algorithm='kdtree', trees=4)

    def find_nearest(self, des):
        result, dist = self.flann.nn_index(des, num_neighbors=1)

        return result

    def add_keyframe_without_ba(self, keyframe, verbose=False):
        keyframe.convert_keypoint_to_array(norm=False)
        super(NNBasedMap, self).add_keyframe_without_ba(keyframe, verbose)

        camera = PTZCamera(keyframe.principal_point, keyframe.camera_center, keyframe.base_rotation)
        rays = camera.back_project_to_rays(keyframe.feature_pts)

        self.global_ray = np.row_stack([self.global_ray, rays])
        self.global_des = np.row_stack([self.global_des, keyframe.feature_des])

    def relocalize(self, keyframe):
        index = self.find_nearest(keyframe.feature_des)

        pass


def ut_pyflann():
    dataset = np.array(
        [[1., 1, 1, 2, 3],
         [10, 10, 10, 3, 2],
         [100, 100, 2, 30, 1]
         ])
    testset = np.array(
        [[1., 1, 1, 1, 1],
         [90, 90, 10, 10, 1]
         ])

    flann = pyflann.FLANN()

    params = flann.build_index(dataset, algorithm='kdtree', trees=4)

    result = flann.nn_index(testset, num_neighbors=1)
