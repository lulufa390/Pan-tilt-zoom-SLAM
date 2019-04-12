//
//  rf_map_builder.hpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-03-30.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#ifndef rf_map_builder_hpp
#define rf_map_builder_hpp

#include <stdio.h>
#include <Eigen/Dense>
#include <string>
#include "bt_dt_regressor.h"
#include "btdtr_ptz_util.h"


class RFMapBuilder {
    using TreeParameter = btdtr_ptz_util::PTZTreeParameter;
    using TreeType = BTDTRTree;
    typedef TreeType* TreePtr;
    
private:
    TreeParameter tree_param_;
    
public:
    RFMapBuilder();
    ~RFMapBuilder();
    
    
    void setTreeParameter(const TreeParameter& param);
    
    // build model from subset of images    
    // sift feature are precomputed to save time
    // feature_label_files: .mat file has ptz, keypoint location and descriptor
    bool buildModel(BTDTRegressor& model,
                    const vector<string> & feature_label_files,
                    const char *model_file_name) const;
    
    // Add one tree to the init model
    // using files from all feature_label_files and part of init_feature_label_files
    // model: input and output   
    // feature_label_files: new added files
    // model_file_name: output, new model
    bool addTree(BTDTRegressor& model,
                const vector<string> & feature_label_files,
                const char *model_file_name);
    
private:    
    bool validationError(const BTDTRegressor & model,
                         const vector<string> & ptz_keypoint_descriptor_files,
                         const int sample_frame_num = 10) const;
    
    // output: selected_indices, examples used to create a new tree
    // selected examples that have large out-of-bag errors
    void outOfBagSampling(const BTDTRegressor & model,
                           vector<VectorXf>& features,
                           vector<VectorXf>& labels,
                           vector<unsigned int> & selected_indices,
                           float feature_dist_threshold = 0.05,
                           float out_of_bag_error_threshold = 0.02);

    
};


#endif /* rf_map_builder_hpp */
