//
//  bundle_adjustment_python.cpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-04-23.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "bundle_adjustment_python.hpp"

extern "C" {
    EXPORTIT void bundle_adjustment_opt(int n_pose, int n_landmark, int n_residual,
                                        const char* file_name, int u, int v,
                                        double * optimized_ptzs,   // output
                                        double * optimized_landmarks) // output
    {
        
    }
    
}
