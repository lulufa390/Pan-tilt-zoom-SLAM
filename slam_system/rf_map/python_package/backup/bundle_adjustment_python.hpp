//
//  bundle_adjustment_python.hpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-04-23.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef bundle_adjustment_python_hpp
#define bundle_adjustment_python_hpp

#include <stdio.h>
#ifdef _WIN32
#define EXPORTIT __declspec( dllexport )
#else
#define EXPORTIT
#endif

extern "C" {
    // file_name: .mat file has, keypoints, src_pt_index, dst_pt_index, landmark_index, camera center and base_rotation
    EXPORTIT void bundle_adjustment_opt(int n_pose, int n_landmark, int n_residual,
                                        const char* file_name, int u, int v,
                                        double * optimized_ptzs,   // output
                                        double * optimized_landmarks); // output
    
    
}


#endif /* bundle_adjustment_python_hpp */
