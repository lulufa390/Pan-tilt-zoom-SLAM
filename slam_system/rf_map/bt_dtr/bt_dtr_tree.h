//
//  Created by jimmy on 2016-12-29.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __BT_DTR_Tree__
#define __BT_DTR_Tree__

// back tracking decision tree Node for regression
// idea: during the testing, back tracking trees once the testing example reaches the leaf node. It "should" increase performance
// disadvantage: increasing storage of the model, decrease speed in testing

#include <vector>
#include <Eigen/Dense>
#include <algorithm>
#include "bt_dtr_util.h"

#include "flann/util/heap.h"
#include "flann/util/result_set.h"
#include <flann/flann.hpp>

using std::vector;
using Eigen::VectorXf;

using flann::BranchStruct;

class BTDTRNode;

class BTDTRTree
{
    friend class BTDTRegressor;
    
    typedef flann::L2<float> Distance;
    typedef Distance::ResultType DistanceType;
    typedef Distance::ElementType ElementType;
    
    typedef BTDTRNode* NodePtr;
    typedef BTDTRTreeParameter TreeParameter;
    typedef BranchStruct<NodePtr, DistanceType > BranchSt;
    typedef BranchSt* Branch;

    
    NodePtr root_;
    TreeParameter tree_param_;
    
    Distance distance_;   // the distance functor
    int leaf_node_num_;   // total leaf node number
    vector<NodePtr> leaf_nodes_;   // leaf node for back tracking    
    
    vector<int> dims_;             // candidate split dimension, only used in training
    
public:
    BTDTRTree();
    ~BTDTRTree(){;}
    
    BTDTRTree(const BTDTRTree & other);
    
    // features:
    // labels: regression label
    // indices:
    bool buildTree(const vector<VectorXf> & features,
                   const vector<VectorXf> & labels,
                   const vector<unsigned int> & indices,
                   const BTDTRTreeParameter & param);
    
    bool predict(const Eigen::VectorXf & feature,
                 const int maxCheck,
                 Eigen::VectorXf & pred) const;
    
    bool predict(const Eigen::VectorXf & feature,
                 const int maxCheck,
                 VectorXf & pred,
                 float & dist);
    
    // each row is a descriptor
    void getLeafNodeDescriptor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    void setLeafNodeDescriptor(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    
    const BTDTRTreeParameter & getTreeParameter(void) const;
    void setTreeParameter(const BTDTRTreeParameter & param);   
    
private:
    // split node into left and right subtree
    bool configureNode(const vector<VectorXf> & features,
                       const vector<VectorXf> & labels,
                       const vector<unsigned int> & indices,
                       BTDTRNode * node);
    
    // record leaf node in an array for O(1) access
    void hashLeafNode();
    
    // set leaf node
    void setLeafNode(const vector<VectorXf> & features,
                     const vector<VectorXf> & labels,
                     const vector<unsigned int> & indices,
                     BTDTRNode * node);
    
    void recordLeafNodes(const NodePtr node, vector<NodePtr> & leafNodes, int & leafNodeIndex);
    
    
    // searchLevel from flann kd tree
    void searchLevel(flann::ResultSet<DistanceType>  & result_set, const ElementType* vec, NodePtr node,
                     const DistanceType min_dist, int & checkCount, const int maxCheck, const float epsError,
                     flann::Heap<BranchSt>* heap, flann::DynamicBitset& checked) const;
    
};


#endif /* defined(__RGBD_RF__BTDTRTree__) */
