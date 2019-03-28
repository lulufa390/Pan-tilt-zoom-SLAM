//
//  gl_ray_3d.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-04.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __CalibMeMatching__gl_ray_3d__
#define __CalibMeMatching__gl_ray_3d__

#include <stdio.h>
#include <Eigen/Dense>

namespace cvx_gl {
    using Eigen::Vector3d;
    class ray_3d {
        Eigen::Vector3d p0_;  // The ray origin
        Eigen::Vector3d t_;   // ray direction vector
        
    public:
        ray_3d(){};
        
        //: Construct from orign and direction
        inline ray_3d(Vector3d const& p0,
                          Vector3d const& direction)
        : p0_(p0), t_(direction) {t_.normalize();}
        
        //: Construct from two points
       // inline gl_ray_3d(Vector3d const& origin,
         //                 Vector3d const& p)
        //: p0_(origin), t_(p-origin) {t_.normalize();}
        
        //: Accessors
        inline Vector3d origin() const { return p0_; } // return a copy
        
        inline Vector3d direction() const
        { return t_/(t_.norm()); } // return a copy        
        
        
        
    };
}

#endif /* defined(__CalibMeMatching__gl_ray_3d__) */
