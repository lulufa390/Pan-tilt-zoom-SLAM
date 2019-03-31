cmake_minimum_required (VERSION 2.6)

cmake_policy(SET CMP0015 NEW)

project (rf_map_lib)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++11")
endif()

# The version number
set (rf_map_lib_MAJOR 1)
set (rf_map_lib_MINOR 0)

set (USE_ANACONDA 1)

include_directories("${PROJECT_BINARY_DIR}")

set(ANACONDA_DIR /Users/jimmy/anaconda3)

# link Eigen
set(Eigen_DIR ${ANACONDA_DIR}/include/eigen3)
include_directories (${Eigen_DIR})

set(CONDA_HEAD ${ANACONDA_DIR}/include)
set(CONDA_LIB ${ANACONDA_DIR}/lib)

# link matio, flann
include_directories (${CONDA_HEAD})
link_directories (${CONDA_LIB})

# add directory files for head files?
include_directories (./ ./bt_dtr ./cvx_gl ./cvx_pgl
                     ./dt_util ./util)


# .cpp in cvx_gl
set(SOURCE_CVX_GL 
	./cvx_gl/gl_homg_point_2d.cpp
	./cvx_gl/gl_homg_point_3d.cpp
	./cvx_gl/gl_ray_3d.cpp)

# .cpp in cvx_pgl
set(SOURCE_CVX_PGL 
	./cvx_pgl/pgl_calibration_matrix.cpp
	./cvx_pgl/pgl_perspective_camera.cpp
	./cvx_pgl/pgl_proj_camera.cpp
	./cvx_pgl/pgl_ptz_camera.cpp)

# .cpp in bt_dtr
set(SOURCE_BT_DTR
   ./bt_dtr/bt_dt_regressor.cpp
   ./bt_dtr/bt_dtr_node.cpp
   ./bt_dtr/bt_dtr_tree.cpp
   ./bt_dtr/bt_dtr_util.cpp)

# .cpp .cxx in dt_util
set(SOURCE_DT_UTIL
   ./dt_util/dt_param_parser.cpp
   ./dt_util/dt_random.cpp
   ./dt_util/dt_util.cpp
   ./dt_util/dt_util_io.cpp
   ./dt_util/mat_io.cpp
   ./dt_util/vnl_random.cxx
   ./dt_util/yael_io.cpp)

# .cpp in util
set(SOURCE_UTIL
   ./util/eigen_geometry_util.cpp
   ./util/ptz_pose_estimation.cpp
   ./util/btdtr_ptz_util.cpp)



set(SOURCE_CODE rf_map_builder.cpp ${SOURCE_CVX_GL} ${SOURCE_CVX_PGL}
     ${SOURCE_BT_DTR} ${SOURCE_DT_UTIL} ${SOURCE_UTIL} )


# add library
add_library(rf_map SHARED ${SOURCE_CODE})
target_link_libraries(rf_map matio flann)


# for python interface
include_directories (./python_package)
set(SOURCE_RF_MAP_PYTHON ./python_package/rf_map_python.cpp)
add_library(rf_map_python SHARED ${SOURCE_CODE} ${SOURCE_RF_MAP_PYTHON})
target_link_libraries(rf_map_python matio flann)


















