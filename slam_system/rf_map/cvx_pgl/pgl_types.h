//
//  pgl_types.h
//  CalibMeMatching
//
//  Created by jimmy on 2017-08-05.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef CalibMeMatching_pgl_types_h
#define CalibMeMatching_pgl_types_h

#include <Eigen/Dense>

namespace cvx_pgl {
    
    typedef Eigen::Matrix<double, 3, 4> Matrix34d;
    typedef Eigen::Matrix<float, 3, 4>  Matrix34f;
}


#endif
