# random forest as map
import numpy as np
from ctypes import cdll
from ctypes import c_int
from ctypes import c_char_p

#@todo hardcode library
lib = cdll.LoadLibrary('/Users/jimmy/Code/ptz_slam/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/librf_map_python.dylib')

class RFMap:
    """
    The map is saved on files.
    """
    def __init__(self, rf_file_name):
        self.rf_file_name = rf_file_name

    def createMap(self, feature_label_files, tree_param_file):
        """
        :param tree_param_file:
        :param feature_label_files:
        :return:
        """
        fl_file = feature_label_files.encode('utf-8')
        tr_file = tree_param_file.encode('utf-8')
        rf_file = self.rf_file_name.encode('utf-8')
        lib.createMap.argtypes = [c_char_p, c_char_p, c_char_p]
        lib.createMap(fl_file, tr_file, rf_file)


    def updateMap(self, prev_feature_label_files, feature_label_files):
        pass

    def relocalization(self, feature_file):
        pass

def ut_create_map():
    rf_map = RFMap('debug.txt')
    tree_param_file = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/ptz_tree_param.txt'
    featue_label_files = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/train_feature_file.txt'
    rf_map.createMap(featue_label_files, tree_param_file)

if __name__ == '__main__':
    ut_create_map()