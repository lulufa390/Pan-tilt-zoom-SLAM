//
//  online_rf_map_builder.hpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-04-14.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef online_rf_map_builder_hpp
#define online_rf_map_builder_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include <string>
#include "bt_dt_regressor.h"
#include "btdtr_ptz_util.h"


class OnlineRFMapBuilder {
    using TreeParameter = btdtr_ptz_util::PTZTreeParameter;
    using TreeType = BTDTRTree;
    typedef TreeType* TreePtr;
    
private:
    TreeParameter tree_param_;
    
    // feature label files in each tree
    vector<vector<string> > tree_feature_label_files_;
    
public:
    OnlineRFMapBuilder();
    ~OnlineRFMapBuilder();
    
    
    void setTreeParameter(const TreeParameter& param);   
   
    
    
    
    bool addTree(BTDTRegressor& model,
                 const string & feature_label_file,
                 const char *model_file_name,
                 bool verbose = true);
    
    // update the last tree in the model
    // feature_label_file: new added feature label file
    bool updateTree(BTDTRegressor& model,                    
                    const string & feature_label_file,
                    const char *model_file_name,
                    bool vervose = true);
    
    // add a tree or update a tree
    bool isAddTree(const BTDTRegressor & model,
                        const string & feature_label_file,
                        const double error_threshold,
                        const double percentage_threshold);

    
    
private:
    // Add one tree to the init model
    // using files from all feature_label_files and part of init_feature_label_files
    // model: input and output
    // feature_label_files: new added files
    // model_file_name: output, new model
    bool addTree(BTDTRegressor& model,
                 const vector<string> & feature_label_files,
                 const char *model_file_name,
                 bool verbose = true);
    
    bool validationError(const BTDTRegressor & model,
                         const vector<string> & ptz_keypoint_descriptor_files,
                         const int sample_frame_num = 10) const;
    
    void computePredictionError(const BTDTRegressor & model,
                                const string & feature_label_file,
                                vector<float> & prediction_error);
};




#endif /* online_rf_map_builder_hpp */
