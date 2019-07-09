//
//  online_rf_map.hpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-07-06.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef online_rf_map_hpp
#define online_rf_map_hpp

#include <stdio.h>
#include "online_rf_map_builder.hpp"

#ifdef _WIN32
#define EXPORTIT __declspec( dllexport )
#else
#define EXPORTIT
#endif

class OnlineRFMap {
public:
    OnlineRFMapBuilder builder_;
    BTDTRegressor model_;
public:
    OnlineRFMap();
    ~OnlineRFMap();
    
    // create a map from a single feature label file
    // call only once
    void createMap(const char * feature_label_file,
                   const char * model_parameter_file,
                   const char * model_name);
    
    // update a map: may add or update a tree
    void updateMap(const char * feature_label_file,
                   const char * model_name);
    
    
    // relocalize a camera using the model
    // parameter_file: testing parameter
    // pan_tilt_zoom: input output
    void relocalizeCamera(const char* feature_location_file_name,
                          const char* test_parameter_file,
                          double* pan_tilt_zoom);    
};

extern "C" {
    EXPORTIT OnlineRFMap* OnlineRFMap_new();
    
    EXPORTIT void OnlineRFMap_delete(OnlineRFMap* rf_map);
    
    EXPORTIT void createOnlineMap(OnlineRFMap* ol_rf_map,
                                  const char * feature_label_file,
                                  const char * model_parameter_file,
                                  const char * model_name);
    
    EXPORTIT void updateOnlineMap(OnlineRFMap* ol_rf_map,
                                  const char * feature_label_file,
                                  const char * model_name);
    
    EXPORTIT void relocalizeCameraOnline(OnlineRFMap* ol_rf_map,
                                   const char* feature_location_file_name,
                                   const char* test_parameter_file,
                                   double* pan_tilt_zoom);
}

#endif /* online_rf_map_hpp */
