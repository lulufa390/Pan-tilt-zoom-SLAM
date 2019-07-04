//
//  rf_map.hpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-07-03.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef rf_map_hpp
#define rf_map_hpp

#include <stdio.h>
#include "bt_dt_regressor.h"

#ifdef _WIN32
#define EXPORTIT __declspec( dllexport )
#else
#define EXPORTIT
#endif

class RFMap {
private:
    BTDTRegressor model_;
    
public:
    void createMap(const char * feature_label_file,
                   const char * model_parameter_file,
                   const char * model_name);
    
    
    // relocalize a camera using the model
    // parameter_file: testing parameter
    // pan_tilt_zoom: input output
    void relocalizeCamera(const char* feature_location_file_name,
                         const char* test_parameter_file,
                         double* pan_tilt_zoom);    
};


extern "C" {
    EXPORTIT RFMap* RFMap_new();
    
    EXPORTIT void RFMap_delete(RFMap* rf_map);

    EXPORTIT void createMap(RFMap* rf_map,
                            const char * feature_label_file,
                            const char * model_parameter_file,
                            const char * model_name);
    
    EXPORTIT void relocalizeCamera(RFMap* rf_map,
                                 const char* feature_location_file_name,
                                 const char* test_parameter_file,
                                 double* pan_tilt_zoom);  
}



#endif /* rf_map_hpp */
