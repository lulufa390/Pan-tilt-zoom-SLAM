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
#include <vector>

using std::string;
using std::vector;

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
    
    /*
     const vector<vgl_point_2d<double> > &wldPts,
     const vector<vgl_point_2d<double> > &imgPts,
     const vector<vgl_line_3d_2_points<double> > & wldLines,
     const vector<vector<vgl_point_2d<double> > > & imgLinePts,
     const vector<vgl_conic<double> > & wldConics,
     const vector<vgl_point_2d<double>> & imgConicPts,
     */
    // raw gui input from user to a .txt file, for Ice hockey
    bool writeGUIRecord(const string& file_name,
                        const string& image_name,
                        const vector<vgl_point_2d<double> > &wld_pts,
                        const vector<vgl_point_2d<double> > &img_pts,
                        const vector<vgl_line_3d_2_points<double> > & wld_lines,
                        const vector<vgl_line_segment_2d<double> > & img_lines,
                        const vector<vgl_conic<double> > & wld_conics,
                        const vector<vgl_point_2d<double>> & img_conic_pts);
    
    // read gui record from a .txt file
    bool readGUIRecord(const string& file_name,
                       string& image_name,
                       vector<vgl_point_2d<double> > &wld_pts,
                       vector<vgl_point_2d<double> > &img_pts,
                       vector<vgl_line_3d_2_points<double> > & wld_lines,
                       vector<vgl_line_segment_2d<double> > & img_lines,
                       vector<vgl_conic<double> > & wld_conics,
                       vector<vgl_point_2d<double>> & img_conic_pts);
}

#endif /* io_util_hpp */
