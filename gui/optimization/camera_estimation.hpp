//
//  camera_estimation.hpp
//  Annotation
//
//  Created by jimmy on 2019-01-27.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef camera_estimation_hpp
#define camera_estimation_hpp

#include <stdio.h>
#include <vector>
#include <vpgl/vpgl_perspective_camera.h>

using std::vector;


namespace cvx {
    // initial calibration from at least 4 point-to-point correspondences
    static bool init_calib(const vector<vgl_point_2d<double> > &wld_pts,
                           const vector<vgl_point_2d<double> > &img_tts,
                           const vgl_point_2d<double> &principle_point,
                           vpgl_perspective_camera<double> &camera);
    
    // optimize perspective_camera by minimizing projected distance    
    // init_camera: camera from algebra calibration
    static bool optimize_perspective_camera(const vector<vgl_point_2d<double> > & wld_pts,
                                            const vector<vgl_point_2d<double> > & img_pts,
                                            const vpgl_perspective_camera<double> & init_camera,
                                            vpgl_perspective_camera<double> & final_camera);
    
}


#endif /* camera_estimation_hpp */
