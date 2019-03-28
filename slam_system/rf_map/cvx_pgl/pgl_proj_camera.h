//
//  pgl_proj_camera.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-04.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__pgl_proj_camera__
#define __CalibMeMatching__pgl_proj_camera__

#include <stdio.h>
#include <Eigen/Dense>
#include "gl_homg_point_3d.h"
#include "gl_homg_point_2d.h"
#include "pgl_types.h"

namespace cvx_pgl {
    using cvx_gl::homg_point_2d;
    using cvx_gl::homg_point_3d;
    using cvx_pgl::Matrix34d;
    
    class proj_camera {
        
    public:
        proj_camera();
        ~proj_camera();
        
        // ----------------- Projections and Backprojections:------------------------        
        homg_point_2d project( const homg_point_3d& world_point ) const;
        
        //: Projection from base class
        virtual void project(const double x, const double y, const double z, double& u, double& v) const;

        Eigen::JacobiSVD<Eigen::MatrixXd>* svd() const;
        
        const Matrix34d& get_matrix() const{ return P_; }
        
        //: Setters mirror the constructors and return true if the setting was successful.
        // In subclasses these should be redefined so that they won't allow setting of
        // matrices with improper form.
        virtual bool set_matrix( const Matrix34d& new_camera_matrix );
        
    private:
        Matrix34d P_;
        mutable Eigen::JacobiSVD<Eigen::MatrixXd> *cached_svd_;  // always can change even if in a const function
    };
}

#endif /* defined(__CalibMeMatching__pgl_proj_camera__) */
