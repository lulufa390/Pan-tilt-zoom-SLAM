//
//  rf_map_python.hpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-03-30.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef rf_map_python_hpp
#define rf_map_python_hpp

#include <stdio.h>

extern "C" {
    // Create a model from a list of feature_label files
    void createMap(const char * feature_label_file,
                   const char * model_parameter_file,
                   const char * model_name);
    // Extend the map by add a new tree
    // the new tree is built from feature_label files
    void extendMap(const char* model_name,
                  const char* prev_feature_label_file,
                  const char* feature_label_file);
    // relocalize a camera using the model
    // parameter_file: testing parameter
    // pan_tilt_zoom: output
    void relocalizeCamera(const char* model_name,
                         const char* parameter_file,
                          double* pan_tilt_zoom);    
}

#endif /* rf_map_python_hpp */
