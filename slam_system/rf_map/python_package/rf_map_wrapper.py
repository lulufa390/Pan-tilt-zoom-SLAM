# random forest as map
import numpy as np
from ctypes import cdll
from ctypes import c_int
from ctypes import c_void_p
from ctypes import c_char_p
from ctypes import Structure
import platform

system = platform.system()

#@todo hardcode library
if system == "Windows":
    lib = cdll.LoadLibrary('C:/graduate_design/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/x64/Debug/rf_map_python.dll')
else:
    lib = cdll.LoadLibrary('/Users/jimmy/Code/ptz_slam/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/librf_map_python.dylib')

class RFMap:
    def __init__(self, rf_file):
        self.rf_file = rf_file
        self.rf_map = lib.RFMap_new()

    def create_map(self, feature_label_files, tree_param_file):
        """
        :param tree_param_file:
        :param feature_label_files: .mat file has 'keypoint', 'descriptor' and 'ptz'
        :return:
        """
        fl_file = feature_label_files.encode('utf-8')
        tr_file = tree_param_file.encode('utf-8')
        rf_file = self.rf_file.encode('utf-8')
        #lib.createMap.argtypes = [c_int, c_char_p, c_char_p, c_char_p]

        #print(type(self.rf_map))
        lib.createMap(self.rf_map, fl_file, tr_file, rf_file)


def ut_create_map():
    rf_map = RFMap('debug.txt')

    if system == "Windows":
        tree_param_file = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/ptz_tree_param.txt'
        featue_label_files = 'C:/graduate_design/random_forest/two_point_method_world_cup_dataset/train_feature_file.txt'
    else:
        tree_param_file = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/ptz_tree_param.txt'
        featue_label_files = '/Users/jimmy/Code/ptz_slam/dataset/two_point_method_world_cup_dataset/train_feature_file.txt'

    rf_map.create_map(featue_label_files, tree_param_file)


if __name__ == '__main__':
    ut_create_map()
