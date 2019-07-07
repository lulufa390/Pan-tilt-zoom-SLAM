# random forest as map
import numpy as np
from ctypes import cdll
from ctypes import c_int
from ctypes import c_void_p
from ctypes import c_char_p
import platform

system = platform.system()

#@todo hardcode library
if system == "Windows":
    lib = cdll.LoadLibrary('C:/graduate_design/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/x64/Debug/rf_map_python.dll')
else:
    lib = cdll.LoadLibrary('/Users/jimmy/Code/ptz_slam/Pan-tilt-zoom-SLAM/slam_system/rf_map/build/librf_map_python.dylib')

class OnlineRFMap:
    """
    online random forest for mapping
    """
    def __init__(self, rf_file_name, tree_param_file):
        self.rf_file_name = rf_file_name
        self.tree_param_file = tree_param_file

        self.feature_label_files = []
        self.feature_label_file_index_each_tree = []


    def createMap(self, feature_label_files, tree_param_file):
        """
        :param tree_param_file: only one tree
        :param feature_label_files: .mat file has 'keypoint', 'descriptor' and 'ptz'
        :return:
        """
        fl_file = feature_label_files.encode('utf-8')
        tr_file = tree_param_file.encode('utf-8')
        rf_file = self.rf_file_name.encode('utf-8')
        lib.createMap.argtypes = [c_char_p, c_char_p, c_char_p]
        lib.createMap(fl_file, tr_file, rf_file)

        self.feature_label_files = feature_label_files
        self.feature_label_file_index_each_tree.append(feature_label_files)


    def addKeyframe(self, feature_label_file):
        pass
        # prediction error percentage from previous model
        #
        # add a tree

        # update a tree

    def relocalization(self, feature_location_file):
        """
        :param feature_file: .mat file has 'keypoint' and 'descriptor'
        :return:
        """
        model_name = self.rf_file_name.encode('utf-8')
        feature_location_file = feature_location_file.encode('utf-8')
        test_parameter_file = ''.encode('utf-8')
        pan_tilt_zoom = np.zeros((3, 1))
        lib.relocalizeCamera.argtrypes = [c_char_p, c_char_p, c_char_p, c_void_p]
        lib.relocalizeCamera(model_name,
                             feature_location_file,
                             test_parameter_file,
                             c_void_p(pan_tilt_zoom.ctypes.data))
        return pan_tilt_zoom