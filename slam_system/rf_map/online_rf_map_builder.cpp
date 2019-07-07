//
//  online_rf_map_builder.cpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-04-14.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "online_rf_map_builder.hpp"
#include "dt_util.hpp"
#include <iostream>
#include <unordered_set>
#include "mat_io.hpp"

using namespace::std;

OnlineRFMapBuilder::OnlineRFMapBuilder()
{
    
}

OnlineRFMapBuilder::~OnlineRFMapBuilder()
{
    
}

void OnlineRFMapBuilder::setTreeParameter(const TreeParameter& param)
{
    tree_param_ = param;
    tree_param_.base_tree_param_.tree_num_ = 0; // initialization
}

bool OnlineRFMapBuilder::addTree(BTDTRegressor& model,
                              const string & feature_label_file,
                              const char *model_file_name,
                              bool verbose)
{
    // 1. get unique files
    unordered_set<string> all_files;
    for (const auto& files: tree_feature_label_files_) {
        for (const string& f:files) {
            all_files.insert(f);
        }
    }
    vector<string> unique_files;
    for (const auto&f: all_files) {
        unique_files.push_back(f);
    }
    
    // 2. sample training files
    const int frame_num = (int)unique_files.size();
    const int sampled_frame_num = std::min(frame_num, tree_param_.sampled_frame_num_) - 1;
    vector<string> sampled_files;
    for (int j = 0; j<sampled_frame_num; j++) {
        int index = rand()%frame_num;
        sampled_files.push_back(unique_files[index]);
    }
    sampled_files.push_back(feature_label_file);
    
    return this->addTree(model, sampled_files, model_file_name, verbose);
}

bool OnlineRFMapBuilder::addTree(BTDTRegressor& model,
                              const vector<string> & feature_label_files,
                              const char *model_file_name,
                              bool verbose)
{
    assert(feature_label_files.size() > 0);
    assert(model.trees_.size() == tree_feature_label_files_.size());
    
    // book keep feature label files
    tree_param_.base_tree_param_.tree_num_ += 1;
    tree_feature_label_files_.push_back(feature_label_files);
    model.reg_tree_param_ = tree_param_.base_tree_param_;
    
    // 1. read training examples
    const Eigen::Vector2f pp(tree_param_.pp_x_, tree_param_.pp_y_);
    vector<VectorXf> features;
    vector<VectorXf> labels;
    for (int j = 0; j<feature_label_files.size(); j++) {
        vector<btdtr_ptz_util::PTZTrainingSample> samples;
        Eigen::Vector3f dummy_ptz;  // not used
        btdtr_ptz_util::generatePTZSampleWithFeature(feature_label_files[j].c_str(), pp, dummy_ptz, samples);
        for (int k = 0; k< samples.size(); k++) {
            features.push_back(samples[k].descriptor_);
            labels.push_back(samples[k].pan_tilt_);
        }
    }
    
    assert(features.size() == labels.size());
    
    // 2. train tree
    vector<unsigned int> indices = DTUtil::range<unsigned int>(0, (int)features.size(), 1);
    model.feature_dim_ = (int)features[0].size();
    model.label_dim_   = (int)labels[0].size();
    
    TreePtr pTree = new TreeType();
    assert(pTree);
    double tt = clock();
    pTree->buildTree(features, labels, indices, tree_param_.base_tree_param_);
    
    
    if (verbose) {
        printf("build a tree cost %lf seconds\n", (clock()-tt)/CLOCKS_PER_SEC );
    }
    
    // 3. update model
    model.trees_.push_back(pTree);
    assert(model.trees_.size() == model.reg_tree_param_.tree_num_);
    
    if (model_file_name != NULL) {
        model.saveModel(model_file_name);
        if (verbose) {
            printf("saved %s\n", model_file_name);
        }
    }
    
    // 4. training error
    if (verbose) {
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
    }
    
    
    //this->validationError(model, feature_label_files, 1);
    return true;
}

bool OnlineRFMapBuilder::updateTree(BTDTRegressor& model,
                                    
                                    const string & feature_label_file,
                                    const char *model_file_name,
                                    bool verbose)
{
    assert(model.trees_.size() > 0);
    
    const int tree_index = rand()%model.trees_.size();
    
    // add new file to book-keeper
    tree_feature_label_files_[tree_index].push_back(feature_label_file);
    const int tree_num = model.treeNum();
    assert(tree_index < tree_num);
    
    // 1. read training examples
    const Eigen::Vector2f pp(tree_param_.pp_x_, tree_param_.pp_y_);
    vector<VectorXf> features;
    vector<VectorXf> labels;
    for (int j = 0; j<tree_feature_label_files_[tree_index].size(); j++) {
        vector<btdtr_ptz_util::PTZTrainingSample> samples;
        Eigen::Vector3f dummy_ptz;  // not used
        btdtr_ptz_util::generatePTZSampleWithFeature(tree_feature_label_files_[tree_index][j].c_str(),
                                                     pp, dummy_ptz, samples);
        for (int k = 0; k< samples.size(); k++) {
            features.push_back(samples[k].descriptor_);
            labels.push_back(samples[k].pan_tilt_);
        }
    }
    assert(features.size() == labels.size());
    
    // 1. update the tree
    vector<unsigned int> indices = DTUtil::range<unsigned int>(0, (int)features.size(), 1);
    assert(indices.size() == features.size());
    model.feature_dim_ = (int)features[0].size();
    model.label_dim_   = (int)labels[0].size();
    
    TreePtr pTree = model.trees_[tree_index];
    assert(pTree);
    double tt = clock();
    pTree->updateTree(features, labels, indices, tree_param_.base_tree_param_);
    printf("update a tree cost %lf seconds\n", (clock()-tt)/CLOCKS_PER_SEC );
    
    if (model_file_name != NULL) {
        model.saveModel(model_file_name);
        if (verbose) {
            printf("saved %s\n", model_file_name);
        }
    }    
    return true;
}

bool OnlineRFMapBuilder::validationError(const BTDTRegressor & model,
                                   const vector<string> & ptz_keypoint_descriptor_files,
                                   const int sample_frame_num) const
{
    const Eigen::Vector2f pp(tree_param_.pp_x_, tree_param_.pp_y_);
    
    const int max_check = 4;
    // sample from selected frames
    for (int i = 0; i<sample_frame_num; i++) {
        int index = rand()%sample_frame_num;
        string feature_file_name = ptz_keypoint_descriptor_files[index];
        vector<btdtr_ptz_util::PTZTrainingSample> samples;
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

bool OnlineRFMapBuilder::isAddTree(const BTDTRegressor & model,
                                         const string & feature_label_file,
                                         const double error_threshold,
                                         const double percentage_threshold)
{
    vector<float> errors;
    this->computePredictionError(model, feature_label_file, errors);
    
    double ratio = 0;
    for (const auto& e:errors) {
        if (e < error_threshold) {
            ratio += 1;
        }
    }
    ratio /= errors.size();
    printf("error threahold: %.03f percentage %.03f\n", error_threshold, ratio);
    if (ratio < percentage_threshold) {
        return true;
    }
    else {
        return false;
    }
}

void OnlineRFMapBuilder::computePredictionError(const BTDTRegressor & model,
                                               const string & feature_label_file,
                                               vector<float> & prediction_error)
{
    // 1. read training examples
    const Eigen::Vector2f pp(tree_param_.pp_x_, tree_param_.pp_y_);
    vector<VectorXf> features;
    vector<VectorXf> labels;
    vector<btdtr_ptz_util::PTZTrainingSample> samples;
    
    Eigen::Vector3f dummy_ptz;  // not used
    btdtr_ptz_util::generatePTZSampleWithFeature(feature_label_file.c_str(), pp, dummy_ptz, samples);
    for (const auto &s:samples) {
        features.push_back(s.descriptor_);
        labels.push_back(s.pan_tilt_);
    }    
    assert(features.size() == labels.size());
    
    // use a pre-trained the model to select new examples
    // if the prediction error is smaller than a threshold, then the new example is discarded
    const int max_check = 4;
    for (int i = 0; i<features.size(); i++) {
        vector<VectorXf> preds;
        vector<float> dists;
        bool is_pred = model.predict(features[i], max_check, preds, dists);
        assert(is_pred);
        VectorXf dif = labels[i] - preds[0];
        float pred_error = dif.norm();
        prediction_error.push_back(pred_error);
    }
}


