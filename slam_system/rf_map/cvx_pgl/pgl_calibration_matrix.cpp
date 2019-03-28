//
//  cvx_pgl_calibration_matrix.cpp
//  CalibMeMatching
//
//  Created by jimmy on 2017-07-30.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "pgl_calibration_matrix.h"

namespace cvx_pgl {
    
    calibration_matrix::calibration_matrix() :
    focal_length_( 1.0 ),
    principal_point_( Eigen::Vector2d( 0.0, 0.0 ) ),
    x_scale_( 1.0 ),
    y_scale_( 1.0 ),
    skew_( 0 )
    {}
    
    
    calibration_matrix::calibration_matrix(double focal_length,
                                           const Eigen::Vector2d& principal_point,
                                           double x_scale, double y_scale, double skew ) :
    
    focal_length_( focal_length ),
    principal_point_( principal_point ),
    x_scale_( x_scale ),
    y_scale_( y_scale ),
    skew_( skew )
    {
        // Make sure the inputs are valid.
        assert( focal_length != 0 );
        assert( x_scale > 0 );
        assert( y_scale > 0 );
    }
    
    
    /*
    //--------------------------------------
    calibration_matrix::calibration_matrix( const Eigen::Matrix3d& K )
    {
        // Put the supplied matrix into canonical form and check that it could be a
        // calibration matrix.
        assert( K(2,2) != 0 && K(1,0) == 0 && K(2,0) == 0.0 && K(2,1) == 0.0 );
        double scale_factor = 1.0;
        if ( K(2,2) != 1.0 ) scale_factor /= (double)K(2,2);
        
        focal_length_ = 1.0;
        x_scale_ = scale_factor*K(0,0);
        y_scale_ = scale_factor*K(1,1);
        skew_    = scale_factor*K(0,1);
        principal_point_ = Eigen::Vector2d( (scale_factor*K(0,2)), (scale_factor*K(1,2)) );
        
        assert( ( x_scale_ > 0 && y_scale_ > 0 ) || ( x_scale_ < 0 && y_scale_ < 0 ) );
    }
     */
    
    Eigen::Matrix3d calibration_matrix::get_matrix() const
    {
        // Construct the matrix as in H&Z.
        Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
        K(0,0) = focal_length_*x_scale_;
        K(1,1) = focal_length_*y_scale_;
        K(2,2) = 1;
        K(0,2) = principal_point_.x();
        K(1,2) = principal_point_.y();
        K(0,1) = skew_;
        return K;
    }

    
    
    
} // namespace