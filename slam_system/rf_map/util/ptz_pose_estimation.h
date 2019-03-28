//
//  ptz_pose_estimation.h
//  PTZBTRF
//
//  Created by jimmy on 2017-08-09.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PTZBTRF__ptz_pose_estimation__
#define __PTZBTRF__ptz_pose_estimation__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>

using std::vector;
using Eigen::Vector2d;
using Eigen::Vector2d;

// optimize pan, tilt and zoom camera pose given
// noise observation
namespace ptz_pose_opt {
    struct PTZPreemptiveRANSACParameter
    {
        double reprojection_error_threshold_;    // distance threshod, unit pixel
        int sample_number_;
   
        PTZPreemptiveRANSACParameter()
        {
            reprojection_error_threshold_ = 2.0; //
            sample_number_ = 32;
        }
    };
    
    ///image_points: image coordinate locations
    // candidate_pan_tilt: corresonding pan, tilt in camera coordinate, have outliers, multiple choices
    // param: RANSAC parameter
    // principal_point: image center
    // ptz: output, camera pan, tilt and focal length
    bool preemptiveRANSACOneToMany(const vector<Eigen::Vector2d> & image_points,
                                   const vector<vector<Eigen::Vector2d> > & candidate_pan_tilt,
                                   const Eigen::Vector2d& principal_point,
                                   const PTZPreemptiveRANSACParameter & param,
                                   Eigen::Vector3d & ptz,
                                   bool verbose = true);
}

#endif /* defined(__PTZBTRF__ptz_pose_estimation__) */
