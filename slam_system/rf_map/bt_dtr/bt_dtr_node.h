//  Created by jimmy on 2016-12-29.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __BT_DTR_Node__
#define __BT_DTR_Node__

// back tracking decision tree Node for regression
// idea: during the testing, back tracking trees once the testing example reaches the leaf node.
// It "should" increase performance
// disadvantage: increasing storage of the model, decrease speed in testing

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include "bt_dtr_util.h"

using std::vector;
using Eigen::VectorXf;


// tree node structure in a tree
class BTDTRNode
{
    typedef BTDTRNode  Node;
    typedef BTDTRNode* NodePtr;
    typedef BTDTRSplitParameter SplitParameter;
    
public:
    BTDTRNode *left_child_;    // left child in a tree
    BTDTRNode *right_child_;   // right child in a tree
    int depth_;                // depth in a tree, from 0
    bool is_leaf_;             // indicator if this is a leaf node
    
    // non-leaf node parameter
    SplitParameter split_param_;  // intermedia node data structure
    
    // leaf node parameter
    VectorXf label_mean_;      // label, e.g., 3D location
    VectorXf label_stddev_;    // standard deviation of labels
    VectorXf feat_mean_;       // mean value of local descriptors, e.g., WHT features
    int index_;                // node index, for save/store tree
    
    // auxiliary data
    int sample_num_;            // num of training examples
    double sample_percentage_;  // ratio of training examples from parent node
    
public:
    BTDTRNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_   = depth;
        is_leaf_ = false;
        sample_num_ = 0;
        sample_percentage_ = 0.0;
        index_ = -1;
    }
    
    static bool writeTree(const char *fileName, const NodePtr root, const int leafNodeNum);
    static bool readTree(const char *fileName, NodePtr & root, int &leafNodeNum);
    
private:
    static void writeNode(FILE *pf, const NodePtr node);
    static void readNode(FILE *pf, NodePtr & node);
    
};

#endif /* defined(__BT_DTR_Node__) */
