//
//  rotation_3d.h
//  calib
//
//  Created by jimmy on 2017-07-27.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef calib_rotation_3d_h
#define calib_rotation_3d_h

#include <Eigen/Dense>
#include <Eigen/Geometry>

using Eigen::Quaternion;
using Eigen::Vector3d;
using Eigen::Matrix3d;

namespace cvx_gl {
    // different representative of rotation matrix
    
    class rotation_3d {
    public:
        rotation_3d()
        {
            q_ = Eigen::Quaternion<double>(1, 0, 0, 0);
        }        
        
        rotation_3d( Quaternion<double> const& q ) : q_(q) { q_.normalize(); }
        
        ~rotation_3d()
        {
            
        }
        
        //: Construct from a 3x3 rotation matrix
        explicit rotation_3d( Eigen::Matrix3d const& matrix )
        : q_(matrix) {}
        
        //: Construct from a Rodrigues vector.
        explicit rotation_3d(Eigen::Vector3d const& rvector )
        {
            double mag = rvector.norm();
            Eigen::Quaternion<double> q;
            if (mag > 0.0) {
                Eigen::Vector3d r = rvector/mag;
                Eigen::AngleAxisd aa(mag, r);
                q_ = Eigen::Quaternion<double>(aa);
            }
            else { // identity rotation is a special case
                q_ = Eigen::Quaternion<double>(1, 0, 0, 0);
            }
        }
        
        //: Construct from a Rodrigues vector.
        
        Vector3d as_rodrigues() const
        {
            Eigen::AngleAxisd aa(q_);
            
            double ang = aa.angle();
            if (ang == 0.0) {
                return Eigen::Vector3d::Zero();
            }
            return aa.axis()*(ang);
        }
        
        //: Output the matrix representation of this rotation in 3x3 form.
        Matrix3d as_matrix() const
        {
            return q_.normalized().toRotationMatrix();
        }
        
        //: Output unit quaternion.
        Eigen::Quaternion<double> as_quaternion() const
        {
            return q_;
        }
        
        //: Make the rotation the identity (i.e. no rotation)
        rotation_3d& set_identity() { q_ = Eigen::Quaternion<double>(1, 0, 0, 0); return *this; }
        
    protected:
        Eigen::Quaternion<double> q_;
        
    };
    
} // namespace


#endif
