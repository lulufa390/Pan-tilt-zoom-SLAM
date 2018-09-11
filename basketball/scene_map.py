import numpy as np
from util import overlap_pan_angle

class Map:
    def __init__(self):
        # [N, 2] float64 array
        self.global_ray = []

        # list of KeyFrame object
        self.keyframe_list = []


    def good_new_keyframe(self, ptz, threshold1 = 5, threshold2 = 20, im_width = 1280, verbose = False):
        """
        good or not as a new keyframe, small overlap with all existing keyframes
        :param ptz: array [3] camera pose
        :threshold1, pan angle overlap in degree, lower bound, too small, no shared features
        :threshold2, pan angle overlap in degree, upp bound, an image covers about 30 degrees
        :return: True for good new key frame, false for not
        """
        # check parameter
        assert ptz.shape[0] == 3
        N = len(len(self.keyframe_list))
        if N == 0:
            print('Warning: not existing key frames')
            return False

        map_ptzs = np.zeros((N, 3))
        for i in range(N):
            map_ptzs[i][0] = self.keyframe_list[i].pan
            map_ptzs[i][1] = self.keyframe_list[i].tilt
            map_ptzs[i][2] = self.keyframe_list[i].f

        pan_angle_overlaps = np.zeros(N)
        for i in range(N):
            pan_angle_overlaps[i] = overlap_pan_angle(ptz[2], ptz[0], map_ptzs[i][2], map_ptzs[i][0], im_width)

        if verbose:
            print('candidate key frame overlap: ', pan_angle_overlaps)
        max_overlap = max(pan_angle_overlaps)
        return max_overlap > threshold1 and max_overlap < threshold2




