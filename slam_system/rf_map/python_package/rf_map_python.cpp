//
//  rf_map_python.cpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-03-30.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "rf_map_python.hpp"
#include <string>
#include <vector>
#include <fstream>
#include "rf_map_builder.hpp"
#include "bt_dt_regressor.h"
#include "btdtr_ptz_util.h"

using namespace::std;
extern "C" {
    // Create a model from a list of feature_label files
    void createMap(const char * feature_label_file,
                   const char * model_parameter_file,
                   const char * model_name)
    {
        // 1. read feature label file
        vector<string> feature_files;
        ifstream file(feature_label_file);
        string str;
        while(std::getline(file, str)) {
            feature_files.push_back(str);
        }
        printf("read %lu feature label files\n", feature_files.size());
       
        
        btdtr_ptz_util::PTZTreeParameter tree_param;
        tree_param.readFromFile(model_parameter_file);
        tree_param.printSelf();
        
        RFMapBuilder builder;
        builder.setTreeParameter(tree_param);
        
        BTDTRegressor model;
        builder.buildModel(model, feature_files, model_name);
        model.saveModel(model_name);
    }
    // Extend the map by add a new tree
    // the new tree is built from feature_label files
    void extendMap(const char* model_name,
                   const char* prev_feature_label_file,
                   const char* feature_label_file)
    {
        
    }
    
    // relocalize a camera using the model
    // parameter_file: testing parameter
    // pan_tilt_zoom: output
    void relocalizeCamera(const char* model_name,
                          const char* parameter_file,
                          double* pan_tilt_zoom)
    {   // read model
        BTDTRegressor model;
        bool is_read = model.load(model_name);
        assert(is_read);
        
        // read testing parameter
        
    }
}
