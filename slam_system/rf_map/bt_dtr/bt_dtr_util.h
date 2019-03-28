//
//  BTDTRUtil.h
//  RGBD_RF
//
//  Created by jimmy on 2016-12-29.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__BTDTRUtil__
#define __RGBD_RF__BTDTRUtil__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include <string>

using std::vector;
using std::string;
using Eigen::VectorXf;
using Eigen::VectorXd;

class BTDTRTreeParameter
{
public:
    int tree_num_;               // number of trees
    int max_tree_depth_;         // tree max depth
    int max_balanced_depth_;     // 0 - max_balanced_tree_depth, encourage balanced tree instead of smaller entropy
    int max_sample_num_;         // select randomly sampled points intead of all samples to estimate optimal split value
    
    int min_leaf_node_;          // leaf node  sample number
    int min_split_node_;         // internal node sample number
    
    int candidate_dim_num_;
    int candidate_threshold_num_;    // number of split in [v_min, v_max]
    double min_split_node_std_dev_;  // 0.1 meter
    
    
    bool verbose_;
    bool verbose_leaf_;
    
    BTDTRTreeParameter()
    {
        tree_num_ = 5;
        max_tree_depth_ = 10;
        max_balanced_depth_ = 5;
        max_sample_num_ = 1000;
        
        min_leaf_node_ = 32;
        min_split_node_ = 8;
        
        candidate_dim_num_ = 6;
        candidate_threshold_num_ = 10;        
        min_split_node_std_dev_ = 0.1;       
        
        verbose_ = false;
        verbose_leaf_ = false;
    }
    
    bool readFromFile(const char *fileName)
    {
        FILE *pf = fopen(fileName, "r");
        if (!pf) {
            printf("can not read from %s \n", fileName);
            return false;
        }
        
        this->readFromFile(pf);
        fclose(pf);
        return true;
    }
    
    bool readFromFile(FILE *pf)
    {
        assert(pf);
        
        const int param_num = 11;
        std::unordered_map<std::string, double> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            double val = 0;
            int ret = fscanf(pf, "%s %lf", s, &val);
            if (ret != 2) {
                printf("read tree parameter Error: %s %f\n", s, val);
                assert(ret == 2);
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == param_num);
        
        tree_num_ = (int)imap[string("tree_num")];
        max_tree_depth_ = (int)imap[string("max_tree_depth")];
        max_balanced_depth_ = (int)imap[string("max_balanced_depth")];
        max_sample_num_ = (int)imap[string("max_sample_num")];
        
        
        min_leaf_node_ = (int)imap[string("min_leaf_node")];
        min_split_node_ = (int)imap[string("min_split_node")];
        candidate_dim_num_ = (int)imap[string("candidate_dim_num")];
        candidate_threshold_num_ = (int)imap[string("candidate_threshold_num")];
        min_split_node_std_dev_ = (double)imap[string("min_split_node_std_dev")];
        
        verbose_ = (bool)imap[string("verbose")];
        verbose_leaf_ = (bool)imap[string("verbose_leaf")];
        return true;
    }
    
    bool writeToFile(FILE *pf)const
    {
        assert(pf);
        fprintf(pf, "tree_num %d\n", tree_num_);
        fprintf(pf, "max_tree_depth %d\n", max_tree_depth_);
        fprintf(pf, "max_balanced_depth %d\n", max_balanced_depth_);
        fprintf(pf, "max_sample_num %d\n", max_sample_num_);
        
        fprintf(pf, "min_leaf_node %d\n", min_leaf_node_);
        fprintf(pf, "min_split_node %d\n", min_split_node_);
        
        fprintf(pf, "candidate_dim_num %d\n", candidate_dim_num_);
        fprintf(pf, "candidate_threshold_num %d\n", candidate_threshold_num_);
        fprintf(pf, "min_split_node_std_dev %f\n", min_split_node_std_dev_);
        
        fprintf(pf, "verbose %d\n", (int)verbose_);
        fprintf(pf, "verbose_leaf %d\n\n", (int)verbose_leaf_);
        return true;
    }
    
    void printSelf() const
    {
        writeToFile(stdout);
    }
};

struct BTDTRSplitParameter
{
public:
    int split_dim_;         // dimension in the feature
    float split_threshold_; // a threshold in feature space
    
    // auxilary data
    float split_loss_;    // loss of split, e.g., spatial variance
    BTDTRSplitParameter()
    {
        split_dim_ = 0;
        split_threshold_ = 0;
        split_loss_ = INT_MAX;
    }
};

/*
// move to dt_util.hpp
class BTDTRUtil
{
public:
    template <class T>
    static double spatialVariance(const vector<T> & labels, const vector<unsigned int> & indices);
    
    template <class T>
    static void meanStddev(const vector<T> & labels, const vector<unsigned int> & indices, T & mean, T & sigma);
    
    template <class T>
    static void mean_stddev(const vector<T>& labels, T & mean, T & sigma);
    
    
    template <class T>
    static T mean(const vector<T> & data, const vector<unsigned int> & indices);
    
    template <class T>
    static T mean(const vector<T> & data);
    
    // median: separate channel
    template <class T>
    static void mean_median_error(const vector<T> & errors,
                                  T & mean,
                                  T & median);
    
    static double inbalance_loss(const int leftNodeSize, const int rightNodeSize);
    
};
 */



#endif /* defined(__RGBD_RF__BTDTRUtil__) */
