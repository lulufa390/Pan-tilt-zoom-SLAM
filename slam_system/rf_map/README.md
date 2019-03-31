The code is modified from https://github.com/lood339/two_point_calib
 

Dependent libraries:

conda install -c conda-forge libmatio

conda install -c conda-forge flann

conda install -c conda-forge eigen


Install:
set up 'ANACONDA_DIR' in the CMakeLists.txt

mkdir build

cd build

cmake ../

make -j4

In Mac system, there will be 'librf_map_python.dylib' int the 'build' folder.
This lib file will be used in the python by ctype. 
  
