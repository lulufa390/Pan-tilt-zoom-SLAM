//
//  pgl_ptz_camera.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-04.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__pgl_ptz_camera__
#define __CalibMeMatching__pgl_ptz_camera__

#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include "pgl_perspective_camera.h"

// camera model from "Mimicking Human Camera Operators" from WACV 2015
namespace cvx_pgl {
    using Eigen::Vector2d;
    using Eigen::Vector3d;
    using std::vector;
    using Eigen::MatrixXd;
    class ptz_camera :public perspective_camera {
        Vector2d     pp_;     // principle point
        Vector3d     cc_;     // camera center
        Vector3d     base_rotation_;     // camera base rotation, rodrigues angle
        
        Vector3d     ptz_; // pan, tilt and focal length, angles in degree
    public:
        ptz_camera(){}
        
        // @brief fl = 2000 is an arbitrary number
        ptz_camera(const Vector2d& pp, const Vector3d& cc,
                   const Vector3d& base_rot, double pan = 0, double tilt = 0, double fl = 2000):pp_(pp),
        cc_(cc), base_rotation_(base_rot), ptz_(pan, tilt, fl){}
        
        
        // camera: has same camera center and base rotation
        // O(1)
        bool set_camera(const perspective_camera& camera);
        
        // O(1)
        bool set_ptz(const Vector3d& ptz);
        
        // assume common parameters are fixed.
        // convert general perspective camera to ptz camera
        // wld_pts: nx3 matrix, world coordinate
        // img_pts: nx2 matrix, image coordinate
        // O(n * iteration)
        bool set_camera(const perspective_camera& camera,
                        const MatrixXd & wld_pts,
                        const MatrixXd & img_pts);
        
        
        double pan(void) { return ptz_[0];}
        double tilt(void) { return ptz_[1]; }
        double focal_length(void) { return ptz_[2];}
        Vector3d ptz(void) { return ptz_; }
        
        
        // project pan tilt ray to (x, y)
        Eigen::Vector2d project(double pan, double tilt) const;
        
        // back project an image pixel to a (pan, tilt)
        Eigen::Vector2d back_project(double x, double y) const;
        
        
        // optimize pan, tilt and focal length given world point and image point correspondences
        // wld_pts: n x 3
        // img_pts: n x 2
        static bool estimatePTZWithFixedBasePositionRotation (const MatrixXd & wld_pts,
                                                              const MatrixXd & img_pts,
                                                              const perspective_camera & init_camera,
                                                              const Vector3d & camera_center,
                                                              const Vector3d & rod,
                                                              Vector3d & ptz,
                                                              perspective_camera & estimated_camera);
    };
    
    // pan, tilt: degree
    Eigen::Matrix3d matrixFromPanYTiltX(double pan, double tilt);
    
    Eigen::Vector2d point2PanTilt(const Eigen::Vector2d& pp,
                                  const Eigen::Vector3d& ptz,
                                  const Eigen::Vector2d& point);
    
    Eigen::Vector2d panTilt2Point(const Eigen::Vector2d& pp,
                                  const Eigen::Vector3d& ptz,
                                  const Eigen::Vector2d& pan_tilt);
    
    // optimize pan, tilt and focal length by minimizing re-projection error
    // return: mean reprojection error
    double optimizePTZ(const Eigen::Vector2d & pp,
                       const vector<Eigen::Vector2d> & pan_tilt,
                       const vector<Eigen::Vector2d> & image_point,
                       const Vector3d& init_ptz,
                       Vector3d & opt_ptz);
    
}


#endif /* defined(__CalibMeMatching__pgl_ptz_camera__) */
