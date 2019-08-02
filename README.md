# Pan-tilt-zoom SLAM
Jikai Lu, Jianhui Chen, James J. Little, "Pan-tilt-zoom SLAM for Sports Videos",
accepted at *British Machine Vision Conference* (BMVC) 2019.

This repository contains Python implementation of this paper.
The code is still being constantly updated.

## Abstract
We present an online SLAM system specifically designed to track pan-tilt-zoom
(PTZ) cameras in highly dynamic sports such as basketball and soccer games. In these
games, PTZ cameras rotate very fast and players cover large image areas. To overcome
these challenges, we propose to use a novel camera model for tracking and to use rays as
landmarks in mapping. Rays overcome the missing depth in pure-rotation cameras. We
also develop an online pan-tilt forest for mapping and introduce moving objects (players)
detection to mitigate negative impacts from foreground objects. We test our method on
both synthetic and real datasets. The experimental results show the superior performance
of our method over previous methods for online PTZ camera pose estimation.

## Dependencies

- Python 3.6
    - Numpy 1.11.3
    - OpenCV for Python 3.4.2
    - Scipy 0.18.1

## File Structure

- **slam_system**: main algorithm folder
    - **ptz_slam.py**: main routine for slam algorithm
    - **ptz_camera.py**: camera model
    - key_frame.py: key frame class definition
    - scene_map.py: map class definition
    - relocalization.py: function for keyframe based relocalization
    - bundle_adjustment.py: BA code for mapping
    - image_process.py: image processing functions
    - experiment.py: functions for experiment
    - **rf_map**: random forest library in C++
    - generator: generate synthesized basketball sequence
    - synthesized_court_sequence: preliminary experiments on synthesized basketball sequence
- gui: a image annotation tool in C++
- pre_processing: tools for data pre processing
- synthesized_point_cloud: preliminary experiments on synthesized keypoints on soccer field
- deprecated: preliminary practice code
- writing: visualize for paper writing

  
