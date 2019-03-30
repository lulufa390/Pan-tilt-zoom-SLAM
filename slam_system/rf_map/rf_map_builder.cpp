//
//  rf_map_builder.cpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-03-30.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "rf_map_builder.hpp"
#include "dt_util.hpp"
#include <iostream>
#include "mat_io.hpp"

using namespace::std;

RFMapBuilder::RFMapBuilder()
{
    
}

RFMapBuilder::~RFMapBuilder()
{
    
}

void RFMapBuilder::setTreeParameter(const TreeParameter& param)
{
    tree_param_ = param;
}

bool  RFMapBuilder::buildModel(BTDTRegressor& model,
                             const vector<string> & feature_label_files,
                             const char *model_file_name) const
{
    assert(feature_label_files.size() > 0);
    
    model.trees_.clear();
    
    tree_param_.printSelf();
    model.reg_tree_param_ = tree_param_.base_tree_param_;
    
    const int frame_num = (int)feature_label_files.size();
    const int sampled_frame_num = std::min((int)feature_label_files.size(), tree_param_.sampled_frame_num_);
    const int tree_num = tree_param_.base_tree_param_.tree_num_;
    
    const Eigen::Vector2f pp(tree_param_.pp_x_, tree_param_.pp_y_);
    for (int n = 0; n<tree_num; n++) {
        // randomly sample frames
        vector<string> sampled_files;
        for (int j = 0; j<sampled_frame_num; j++) {
            int index = rand()%frame_num;
            sampled_files.push_back(feature_label_files[index]);
        }
        
        printf("training from %lu frames\n", sampled_files.size());
        // sample from selected frames
        vector<VectorXf> features;
        vector<VectorXf> labels;
        for (int j = 0; j<sampled_files.size(); j++) {
            vector<btdtr_ptz_util::PTZSample> samples;
            Eigen::Vector3f dummy_ptz;  // not used
            btdtr_ptz_util::generatePTZSampleWithFeature(sampled_files[j].c_str(), pp, dummy_ptz, samples);
            for (int k = 0; k< samples.size(); k++) {
                features.push_back(samples[k].descriptor_);
                labels.push_back(samples[k].pan_tilt_);
            }
        }
        assert(features.size() == labels.size());
        
        printf("training sample number is %lu\n", features.size());
        
        model.feature_dim_ = (int)features[0].size();
        model.label_dim_   = (int)labels[0].size();
        
        vector<unsigned int> indices = DTUtil::range<unsigned int>(0, (int)features.size(), 1);
        assert(indices.size() == features.size());
        
        TreePtr pTree = new TreeType();
        assert(pTree);
        double tt = clock();
        pTree->buildTree(features, labels, indices, tree_param_.base_tree_param_);
        printf("build a tree cost %lf seconds\n", (clock()-tt)/CLOCKS_PER_SEC );
        
        // test training error
        vector<Eigen::VectorXf> errors;
        for (int k = 0; k< features.size(); k++) {
            Eigen::VectorXf feat = features[k];
            Eigen::VectorXf label = labels[k];
            Eigen::VectorXf pred;
            float dist = 0.0f;
            pTree->predict(feat, 1, pred, dist);
            errors.push_back(pred - label);
        }
        Eigen::VectorXf q1_error, q2_error, q3_error;
        DTUtil::quartileError(errors, q1_error, q2_error, q3_error);
        cout<<"Training first quartile error: \n"<<q1_error.transpose()<<endl;
        cout<<"Training second quartile (median) error: \n"<<q2_error.transpose()<<endl;
        cout<<"Training third quartile error: \n"<<q3_error.transpose()<<endl<<endl;
                
        model.trees_.push_back(pTree);
        if (model_file_name != NULL) {
            model.saveModel(model_file_name);
            printf("saved %s\n", model_file_name);
        }
        this->validationError(model, feature_label_files, std::min(4, frame_num));
    }
    return true;
}

bool RFMapBuilder::addTree(BTDTRegressor& model,
                           const vector<string> & init_feature_label_files,
                           const vector<string> & feature_label_files,
                           const char *model_file_name)
{
    assert(init_feature_label_files.size() > 0);
    assert(feature_label_files.size() > 0);
    
    tree_param_.base_tree_param_.tree_num_ += 1;
    const int sampled_frame_num = tree_param_.sampled_frame_num_ - (int)feature_label_files.size();
    assert(sampled_frame_num >= 0);
    
    const Eigen::Vector2f pp(tree_param_.pp_x_, tree_param_.pp_y_);
    vector<string> sampled_files = feature_label_files;
    
    // randomly sample frames from previous frames
    for (int j = 0; j<sampled_frame_num; j++) {
        int index = rand()%init_feature_label_files.size();
        sampled_files.push_back(init_feature_label_files[index]);
    }
    
    printf("training from %lu frames\n", sampled_files.size());
    // sample from selected frames
    vector<VectorXf> features;
    vector<VectorXf> labels;
    for (int j = 0; j<sampled_files.size(); j++) {
        vector<btdtr_ptz_util::PTZSample> samples;
        Eigen::Vector3f dummy_ptz;  // not used
        btdtr_ptz_util::generatePTZSampleWithFeature(sampled_files[j].c_str(), pp, dummy_ptz, samples);
        for (int k = 0; k< samples.size(); k++) {
            features.push_back(samples[k].descriptor_);
            labels.push_back(samples[k].pan_tilt_);
        }
    }
    assert(features.size() == labels.size());
    
    printf("training sample number is %lu\n", features.size());
    
    model.feature_dim_ = (int)features[0].size();
    model.label_dim_   = (int)labels[0].size();
    
    vector<unsigned int> indices = DTUtil::range<unsigned int>(0, (int)features.size(), 1);
    assert(indices.size() == features.size());
    
    TreePtr pTree = new TreeType();
    assert(pTree);
    double tt = clock();
    pTree->buildTree(features, labels, indices, tree_param_.base_tree_param_);
    printf("build a tree cost %lf seconds\n", (clock()-tt)/CLOCKS_PER_SEC );
    
    // training error
    vector<Eigen::VectorXf> errors;
    for (int k = 0; k< features.size(); k++) {
        Eigen::VectorXf feat = features[k];
        Eigen::VectorXf label = labels[k];
        Eigen::VectorXf pred;
        float dist = 0.0f;
        pTree->predict(feat, 1, pred, dist);
        errors.push_back(pred - label);
    }
    Eigen::VectorXf q1_error, q2_error, q3_error;
    DTUtil::quartileError(errors, q1_error, q2_error, q3_error);
    cout<<"Training first quartile error: \n"<<q1_error.transpose()<<endl;
    cout<<"Training second quartile (median) error: \n"<<q2_error.transpose()<<endl;
    cout<<"Training third quartile error: \n"<<q3_error.transpose()<<endl<<endl;
    
    model.trees_.push_back(pTree);
    model.reg_tree_param_.tree_num_ += 1;
    assert(model.trees_.size() == model.reg_tree_param_.tree_num_);
    
    if (model_file_name != NULL) {
        model.saveModel(model_file_name);
        printf("saved %s\n", model_file_name);
    }
    
    this->validationError(model, init_feature_label_files, 4);
    return true;
}

bool RFMapBuilder::validationError(const BTDTRegressor & model,
                                   const vector<string> & ptz_keypoint_descriptor_files,
                                   const int sample_frame_num) const
{
    const Eigen::Vector2f pp(tree_param_.pp_x_, tree_param_.pp_y_);
    
    const int max_check = 4;
    // sample from selected frames
    for (int i = 0; i<sample_frame_num; i++) {
        int index = rand()%sample_frame_num;
        string feature_file_name = ptz_keypoint_descriptor_files[index];
        vector<btdtr_ptz_util::PTZSample> samples;
        Eigen::Vector3f dummy_ptz;
        btdtr_ptz_util::generatePTZSampleWithFeature(feature_file_name.c_str(), pp, dummy_ptz, samples);
        
        vector<Eigen::VectorXf> errors;
        vector<float> distance;
        for (int k = 0; k< samples.size(); k++) {
            Eigen::VectorXf feat = samples[k].descriptor_;
            Eigen::VectorXf label = samples[k].pan_tilt_;
            vector<Eigen::VectorXf> cur_predictions;
            vector<float> dist;
            model.predict(feat, max_check, cur_predictions, dist);
            long int min_v_index = std::distance(dist.begin(), std::min_element(dist.begin(), dist.end()));
            
            distance.push_back(dist[min_v_index]);
            errors.push_back(cur_predictions[min_v_index] - label);
            //cout<<"feature distance "<<distance[min_v_index]<<endl;
            //cout<<"prediction: "<<cur_predictions[min_v_index].transpose()<<endl;
            //cout<<"label:      "<<label.transpose()<<endl<<endl;
        }
        
        
        Eigen::VectorXf q1_error, q2_error, q3_error;
        DTUtil::quartileError(errors, q1_error, q2_error, q3_error);
        std::sort(distance.begin(), distance.end());
        printf("tree number: %lu, back tracking number %d\n", model.trees_.size(), max_check);
        cout<<"Validation first quartile error: \n"<<q1_error.transpose()<<endl;
        cout<<"Validation second quartile (median) error: \n"<<q2_error.transpose()<<endl;
        cout<<"Validation third quartile error: \n"<<q3_error.transpose()<<endl;
        cout<<"Validation median feature distance is "<<distance[distance.size()/2]<<endl<<endl;
    }    
    return true;
}
