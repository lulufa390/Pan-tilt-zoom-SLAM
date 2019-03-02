//
//  io_util.hpp
//  CameraCalibration
//
//  Created by jimmy on 2019-02-16.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef io_util_hpp
#define io_util_hpp

#include <stdio.h>
#include <string>
#include <vpgl/vpgl_perspective_camera.h>


namespace io_util {
	// write camera parameter to a .txt file
	// file_name: .txt
	// image_name: corresponding color image
	bool writeCamera(const char *file_name,
		const char *image_name,
		const vpgl_perspective_camera<double> & camera);
	// read a camera and an image name from a .txt file
	bool readCamera(const char *file_name,
		std::string & image_name,
		vpgl_perspective_camera<double> & camera);

	// read court model from a .txt file
	bool readModel(const char * file_name,
		std::vector<vgl_point_2d<double>> & points,
		std::vector<std::pair<int, int>> & index);
}

#endif /* io_util_hpp */
