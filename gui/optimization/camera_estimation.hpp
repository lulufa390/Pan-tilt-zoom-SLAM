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
    bool init_calib(const vector<vgl_point_2d<double> > &wld_pts,
                           const vector<vgl_point_2d<double> > &img_tts,
                           const vgl_point_2d<double> &principle_point,
                           vpgl_perspective_camera<double> &camera);
    
    // optimize perspective_camera by minimizing projected distance    
    // init_camera: camera from algebra calibration
    bool optimize_perspective_camera(const vector<vgl_point_2d<double> > & wld_pts,
                                            const vector<vgl_point_2d<double> > & img_pts,
                                            const vpgl_perspective_camera<double> & init_camera,
                                            vpgl_perspective_camera<double> & final_camera);
    
    
    // initial calibration from point-to-point and line-segment-to-line-segment correspondences
    // image_line_segment: two end point are locations in the image
    // world_line_segment: two end point are locations in the world coordiante, there are Not point-to-point
    // correcpondence
    // First, estimate H, then deompose camera parameters from H
    // assome camera center Z > 0
    bool init_calib(const vector<vgl_point_2d<double> >& wld_pts,
                           const vector<vgl_point_2d<double> >& img_pts,
                           const vector<vgl_line_segment_2d<double>>& world_line_segment,
                           const vector<vgl_line_segment_2d<double>>& image_line_segment,
                           const vgl_point_2d<double> &principlePoint,
                           vpgl_perspective_camera<double> &camera);
    
}


#endif /* camera_estimation_hpp */
