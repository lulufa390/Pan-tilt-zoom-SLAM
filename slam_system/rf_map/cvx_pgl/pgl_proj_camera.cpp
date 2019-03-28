//
//  pgl_proj_camera.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-04.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "pgl_proj_camera.h"

namespace cvx_pgl {
    
    proj_camera::proj_camera()
    {
        cached_svd_ = NULL;
    }
    
    proj_camera::~proj_camera()
    {
        
    }
    
    homg_point_2d proj_camera::project( const homg_point_3d& world_point ) const
    {
        // For efficiency, manually compute the multiplication rather than converting to
        // vnl and converting back.
        homg_point_2d image_point(P_(0,0)*world_point.x() + P_(0,1)*world_point.y() +
                                  P_(0,2)*world_point.z() + P_(0,3)*world_point.w(),
                                  
                                  P_(1,0)*world_point.x() + P_(1,1)*world_point.y() +
                                  P_(1,2)*world_point.z() + P_(1,3)*world_point.w(),
                                  
                                  P_(2,0)*world_point.x() + P_(2,1)*world_point.y() +
                                  P_(2,2)*world_point.z() + P_(2,3)*world_point.w() );
        
        return image_point;
    }
    
    
    void proj_camera::project(const double x, const double y, const double z, double& u, double& v) const
    {
        homg_point_3d world_point(x, y, z);
        homg_point_2d image_point = this->project(world_point);
        if (image_point.ideal(static_cast<double>(1.0e-10)))
        {
            u = 0; v = 0;
            printf("Warning: projection to ideal image point in vpgl_proj_camera - result not valid\n");
            return;
        }
        u = image_point.x()/image_point.w();
        v = image_point.y()/image_point.w();
    }
    
    
    Eigen::JacobiSVD<Eigen::MatrixXd>* proj_camera::svd() const
    {
        // Check if the cached copy is valid, if not recompute it.
        if ( cached_svd_ == NULL )
        {
            cached_svd_ = new Eigen::JacobiSVD<Eigen::MatrixXd>(P_);
            
            // Check that the projection matrix isn't degenerate.
            if ( cached_svd_->rank() != 3 ){
                printf("proj_camera::svd()\n Warning: Projection matrix is not rank 3, errors may occur. \n");
            }
        }
        return cached_svd_;
    }    
    
    bool proj_camera::set_matrix( const Matrix34d& new_camera_matrix )
    {
        P_ = new_camera_matrix;
        if ( cached_svd_ != NULL ) delete cached_svd_;
        cached_svd_ = NULL;
        return true;
    }
}