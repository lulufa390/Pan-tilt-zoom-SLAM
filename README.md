# Camera-Calibration

Camera calibration using python. 

Files:
  1. main.py to draw lines using the ground truth camera pose
  2. composition.py combines two images using their camera pose.
  3. run_synthesize.py generate feature points on 3d soccer field. Output the virtual images using the camera pose on real highlight seq3 dataset. synthesize.py provide some functions for run_synthesiz.py
  4. .mat file is synthesized data.

Update:  
  1. refine_slam.py is the main algorithm. It implements EKF algorithm for PTZ camera calibration. 
