//
//  cvx_vpgl_perspective_camera.cpp
//  calib
//
//  Created by jimmy on 2017-07-26.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "pgl_perspective_camera.h"
#include <Eigen/Geometry>

namespace cvx_pgl {
    perspective_camera::perspective_camera()
    {
        R_.set_identity();
        camera_center_.setConstant(0);
    }
    
    perspective_camera::~perspective_camera()
    {
        
    }
    
    perspective_camera::perspective_camera( const perspective_camera& other )
    {
        if (&other == this) {
            return;
        }
        K_ = other.K_;
        camera_center_ = other.camera_center_;
        R_ = other.R_;
        recompute_matrix();
    }
    
        
    void perspective_camera::set_calibration( const calibration_matrix& K )
    {
        K_ = calibration_matrix(K);
        recompute_matrix();
    }
    void perspective_camera::set_camera_center( const Eigen::Vector3d& camera_center )
    {
        camera_center_ = camera_center;
        recompute_matrix();
    }
    void perspective_camera::set_translation(const Eigen::Vector3d& t)
    {
        Eigen::Matrix3d Rt = R_.as_matrix().transpose();
        Eigen::Vector3d cv = -(Rt * t);
        camera_center_ = cv;
        recompute_matrix();
    }
    
    void perspective_camera::set_rotation(const Eigen::Vector3d& rvector)
    {
        R_ = rotation3d(rvector);
        recompute_matrix();
    }
    
    void perspective_camera::set_rotation( const Eigen::Matrix3d& R )
    {
        R_ = rotation3d(R);
        recompute_matrix();
    }
    
    
    
     void perspective_camera::recompute_matrix()
     {
         // Set new projection matrix to [ I | -C ].
         Matrix34d Pnew;
         Pnew.setConstant(0);
         
         for ( int i = 0; i < 3; i++ ){
             Pnew(i,i) = 1.0;
         }
         Pnew(0,3) = -camera_center_.x();
         Pnew(1,3) = -camera_center_.y();
         Pnew(2,3) = -camera_center_.z();
         
         this->set_matrix(K_.get_matrix() * R_.as_matrix() * Pnew);
     }
    
}