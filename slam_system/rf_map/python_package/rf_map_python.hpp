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

#ifdef _WIN32
	#define EXPORTIT __declspec( dllexport )
#else
	#define EXPORTIT
#endif

extern "C" {
    class BTDTRegressor;
    // Using files for save/load model
    // Create a model from a list of feature_label files
	EXPORTIT void createMap(const char * feature_label_file,
                   const char * model_parameter_file,
                   const char * model_name);
    
    // Extend the map by add a new tree
    // the new tree is built from feature_label files
	EXPORTIT void extendMap(const char* pre_model_name,
                                      const char * model_parameter_file,
                                      const char* feature_label_file,
                                      const char * model_name);
    
    // relocalize a camera using the model
    // parameter_file: testing parameter
    // pan_tilt_zoom: output
	EXPORTIT void relocalizeCamera(const char* model_name,
                         const char* feature_location_file_name,
                         const char* test_parameter_file,
                         double* pan_tilt_zoom);
    
    // Using point for save/load model
    EXPORTIT void buildMap(const char * feature_label_file,
                                   const char * model_parameter_file,
                                     const char * model_name);
}

#endif /* rf_map_python_hpp */
