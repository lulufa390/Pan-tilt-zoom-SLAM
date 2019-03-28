//
//  BTDTRegressor.cpp
//  RGBD_RF
//
//  Created by jimmy on 2016-12-30.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "bt_dt_regressor.h"
#include <string>
#include "bt_dtr_node.h"
#include "yael_io.h"
#include "cvx_util.hpp"

using std::string;

bool BTDTRegressor::predict(const Eigen::VectorXf & feature,
                            const int maxCheck,
                            Eigen::VectorXf & pred) const
{
    assert(trees_.size() > 0);
    assert(feature_dim_ == feature.size());
    
    pred = Eigen::VectorXf::Zero(label_dim_);
    
    // average predictions from all trees
    int pred_num = 0;
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf cur_pred;
        bool is_pred = trees_[i]->predict(feature, maxCheck, cur_pred);
        if (is_pred) {
            pred += cur_pred;
            pred_num++;
        }
    }
    if (pred_num == 0) {
        return false;
    }
    pred /= pred_num;
    return true;
}


bool BTDTRegressor::predict(const Eigen::VectorXf & feature,
                            const int maxCheck,
                            vector<Eigen::VectorXf> & predictions) const
{
    assert(trees_.size() > 0);
    assert(feature_dim_ == feature.size());
    
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf cur_pred;
        bool is_pred = trees_[i]->predict(feature, maxCheck, cur_pred);
        if (is_pred) {
            predictions.push_back(cur_pred);
        }
    }
    return predictions.size() == trees_.size();
}

bool BTDTRegressor::predict(const Eigen::VectorXf & feature,
                            const int maxCheck,
                            vector<Eigen::VectorXf> & predictions,
                            vector<float> & dists) const
{
    assert(trees_.size() > 0);
    assert(feature_dim_ == feature.size());
    assert(predictions.size() == 0);
    assert(dists.size() == 0);
    
    // Step 1: predict from each tree
    vector<Eigen::VectorXf> unordered_predictions;
    vector<float> unordered_dists;
    for (int i = 0; i<trees_.size(); i++) {
        Eigen::VectorXf cur_pred;
        float dist;
        bool is_pred = trees_[i]->predict(feature, maxCheck, cur_pred, dist);
        if (is_pred) {
            unordered_predictions.push_back(cur_pred);
            unordered_dists.push_back(dist);
        }
    }
    assert(unordered_predictions.size() == unordered_dists.size());
    
    // Step 2: ordered by local patch feature distance
    vector<size_t> sortIndexes = CvxUtil::sortIndices<float>(unordered_dists);
    for (int i = 0; i<sortIndexes.size(); i++) {
        predictions.push_back(unordered_predictions[sortIndexes[i]]);
        dists.push_back(unordered_dists[sortIndexes[i]]);
    }
    
    return predictions.size() == trees_.size();
}

bool BTDTRegressor::predict(const Eigen::VectorXf & feature,
                            const int maxCheck,
                            const int maxTreeNum,
                            vector<Eigen::VectorXf> & predictions,
                            vector<float> & dists) const
{
    assert(trees_.size() > 0);
    assert(feature_dim_ == feature.size());
    assert(predictions.size() == 0);
    assert(dists.size() == 0);
    assert(maxTreeNum >= 0 && maxTreeNum <= trees_.size());
    
    // Step 1: predict from each tree
    vector<Eigen::VectorXf> unordered_predictions;
    vector<float> unordered_dists;
    for (int i = 0; i<trees_.size() && i< maxTreeNum; i++) {
        Eigen::VectorXf cur_pred;
        float dist;
        bool is_pred = trees_[i]->predict(feature, maxCheck, cur_pred, dist);
        if (is_pred) {
            unordered_predictions.push_back(cur_pred);
            unordered_dists.push_back(dist);
        }
    }
    assert(unordered_predictions.size() == unordered_dists.size());
    
    // Step 2: ordered by local patch feature distance
    vector<size_t> sortIndexes = CvxUtil::sortIndices<float>(unordered_dists);
    for (int i = 0; i<sortIndexes.size(); i++) {
        predictions.push_back(unordered_predictions[sortIndexes[i]]);
        dists.push_back(unordered_dists[sortIndexes[i]]);
    }
    
    return predictions.size() == maxTreeNum;
}

bool BTDTRegressor::saveModel(const char *file_name) const
{
    assert(trees_.size() > 0);
    // write tree number and tree files to file Name
    FILE *pf = fopen(file_name, "w");
    if(!pf) {
        printf("Error: can not open file %s\n", file_name);
        return false;
    }
    fprintf(pf, "%d %d\n", feature_dim_, label_dim_);
    reg_tree_param_.writeToFile(pf);
    vector<string> tree_files;
    string baseName = string(file_name);
    baseName = baseName.substr(0, baseName.size()-4);
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string fileName = baseName + string(buf) + string(".txt");
        fprintf(pf, "%s\n", fileName.c_str());
        tree_files.push_back(fileName);
    }
    
    // leaf node feature
    vector<string> leaf_node_files;
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "_%08d", i);
        string fileName = baseName + string(buf) + string(".fvec");
        fprintf(pf, "%s\n", fileName.c_str());
        leaf_node_files.push_back(fileName);
    }
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data;
            // get descriptors from leaf node
            trees_[i]->getLeafNodeDescriptor(data);
            YaelIO::write_fvecs_file(leaf_node_files[i].c_str(), data);
        }
    }
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            BTDTRNode::writeTree(tree_files[i].c_str(), trees_[i]->root_, trees_[i]->leaf_node_num_);
        }
    }
    fclose(pf);
    printf("save to %s\n", file_name);
    return true;
}

bool BTDTRegressor::load(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", fileName);
        return false;
    }
    
    int ret_num = fscanf(pf, "%d %d", &feature_dim_, &label_dim_);
    assert(ret_num == 2);
    
    bool is_read = reg_tree_param_.readFromFile(pf);
    assert(is_read);
    reg_tree_param_.printSelf();
    
    // read tree file
    vector<string> treeFiles;
    for (int i = 0; i<reg_tree_param_.tree_num_; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        treeFiles.push_back(string(buf));
    }
    
    // read leaf node descriptor file
    vector<string> leaf_node_files;
    for (int i = 0; i<reg_tree_param_.tree_num_; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        leaf_node_files.push_back(string(buf));
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        if (trees_[i]) {
            delete trees_[i];
            trees_[i] = NULL;
        }
    }
    trees_.clear();
    
    // read each tree
    for (int i = 0; i<treeFiles.size(); i++) {
        BTDTRNode * root = NULL;
        int leaf_node_num = 0;
        bool isRead = false;
        isRead = BTDTRNode::readTree(treeFiles[i].c_str(), root, leaf_node_num);
        assert(isRead);
        assert(root);
        
        BTDTRTree *tree = new BTDTRTree();
        tree->root_ = root;
        tree->setTreeParameter(reg_tree_param_);
        tree->leaf_node_num_ = leaf_node_num;
        
        // read leaf node descriptor and set it in the tree
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data;
        isRead = YaelIO::read_fvecs_file(leaf_node_files[i].c_str(), data);
        assert(is_read);
        tree->setLeafNodeDescriptor(data);
        
        trees_.push_back(tree);
    }
    printf("read from %s\n", fileName);
    return true;
    
}