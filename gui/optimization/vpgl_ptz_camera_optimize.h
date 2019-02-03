//
//  vpgl_ptz_camera_optimize.h
//  QuadCopter
//
//  Created by jimmy on 6/29/15.
//  Copyright (c) 2015 Nowhere Planet. All rights reserved.
//

#ifndef __QuadCopter__vpgl_ptz_camera_optimize__
#define __QuadCopter__vpgl_ptz_camera_optimize__

#include "vpgl_ptz_camera.h"

struct VpglPTZCameraOptimizeParameter
{
    double outlier_threshold_;    // outlier will not be included in optimization
    double line_point_weight_;
    double conic_point_weight_;
    
    VpglPTZCameraOptimizeParameter()
    {
        outlier_threshold_ = 200.0;
        line_point_weight_  = 1.0;
        conic_point_weight_ = 1.0;
    }
};
// optimize ptz camera by multiple constraint
class VpglPTZCameraOptimize
{
public:
    // iterative cloesest points (ICP) by line segments and conic (only circle)
    static bool optimize_PTZ_camera_ICP(const vcl_vector<vgl_point_3d<double> > &wldPts,
                                                const vcl_vector<vgl_point_2d<double> > &imgPts,
                                                const vcl_vector<vgl_line_3d_2_points<double> > & wldLines,
                                                const vcl_vector<vcl_vector<vgl_point_2d<double> > > & imgLinePts,
                                                const vcl_vector<vgl_conic<double> > & wldConics,
                                                const vcl_vector<vcl_vector<vgl_point_2d<double> > > & imgConicPts,
                                                const vpgl_ptz_camera & initPTZ,
                                                const VpglPTZCameraOptimizeParameter & para,
                                                vpgl_ptz_camera & optimizedPTZ,
                                                vpgl_perspective_camera<double> & camera);
    
};

#endif /* defined(__QuadCopter__vpgl_ptz_camera_optimize__) */
