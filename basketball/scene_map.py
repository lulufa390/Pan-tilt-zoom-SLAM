import numpy as np


class Map:
    def __init__(self):
        # [N, 2] float64 array
        self.global_ray = []

        # list of KeyFrame object
        self.keyframe_list = []


    def is_new_key_frame(self, camera_pose):
        """
        :param camera_pose: array [3] camera pose
        :return: True for good new key frame, false for not
        """
        for i in range(len(self.keyframe_list)):
            #@todo: check the angle difference with all keyframes