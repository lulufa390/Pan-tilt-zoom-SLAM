//  Created by jimmy on 2016-12-29.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "bt_dtr_tree.h"
#include "bt_dtr_node.h"
#include "bt_dtr_util.h"
#include "dt_util.hpp"
#include <iostream>
#include "dt_random.hpp"



using std::cout;
using std::endl;


BTDTRTree::BTDTRTree()
{
    root_ = NULL;
    leaf_node_num_ = 0;
}

BTDTRTree::BTDTRTree(const BTDTRTree & other)
{
    // shallow copy
    root_ = other.root_;
    tree_param_ = other.tree_param_;
    leaf_node_num_ = other.leaf_node_num_;
    
    std::copy(other.leaf_nodes_.begin(), other.leaf_nodes_.end(), leaf_nodes_.begin());
}

bool BTDTRTree::buildTree(const vector<VectorXf> & features,
               const vector<VectorXf> & labels,
               const vector<unsigned int> & indices,
               const BTDTRTreeParameter & param)
{
    assert(features.size() == labels.size());
    assert(indices.size() <= features.size());
    
    tree_param_ = param;
    root_ = new BTDTRNode(0);
    leaf_node_num_ = 0;
    
    
    for (unsigned int i = 0; i<features.front().size(); i++) {
        dims_.push_back(i);
    }
    
    // build tree
    this->configureNode(features, labels, indices, root_);
    
    // record leaf node
    this->hashLeafNode();
    
    return true;
}


static bool bestSplitDimension(const vector<VectorXf> & features,
                               const vector<VectorXf> & labels,
                               const vector<unsigned int> & indices,
                               const BTDTRTreeParameter & tree_param,
                               const int depth,
                               BTDTRSplitParameter & split_param,
                               vector<unsigned int> & left_indices,
                               vector<unsigned int> & right_indices)
{
    // randomly select number in a range
    const int dim = split_param.split_dim_;
    double min_v = std::numeric_limits<double>::max();
    double max_v = std::numeric_limits<double>::min();
    const int threshold_num = tree_param.candidate_threshold_num_;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        double v = features[index][dim];
        if (v > max_v) {
            max_v = v;
        }
        if (v < min_v) {
            min_v = v;
        }
    }
    if (!(min_v < max_v)) {
        return false;
    }
    
    vector<double> rnd_split_values = DTRandom::generateRandomNumber(min_v, max_v, threshold_num);
    
    bool is_use_balance = false;
    if (depth <= tree_param.max_balanced_depth_) {
        is_use_balance = true;
    }
    
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    const int min_split_num = tree_param.min_split_node_;
    for (int i = 0; i<rnd_split_values.size(); i++) {
        double threshold = rnd_split_values[i];
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        // split data by comparing with the threshold
        for (int j = 0; j<indices.size(); j++) {
            int index = indices[j];
            double v = features[index][dim];
            if (v < threshold) {
                cur_left_indices.push_back(index);
            }
            else {
                cur_right_indices.push_back(index);
            }
        }
        
        if (cur_left_indices.size() < min_split_num ||
            cur_right_indices.size() < min_split_num) {
            //printf("lefe sample number %lu, right sample number %lu\n", cur_left_indices.size(), cur_right_indices.size());
            continue;
        }
        
        double cur_loss = 0.0;
        if (is_use_balance) {
            cur_loss += DTUtil::balanceLoss((int)cur_left_indices.size(), (int)cur_right_indices.size());
        } else {
            cur_loss += DTUtil::spatialVariance<VectorXf>(labels, cur_left_indices);
            cur_loss += DTUtil::spatialVariance<VectorXf>(labels, cur_right_indices);
        }
        
        if (cur_loss < loss) {
            loss = cur_loss;
            is_split = true;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
            split_param.split_threshold_ = threshold;
            split_param.split_loss_ = cur_loss;
        }
    }
    
    return is_split;
}


bool BTDTRTree::configureNode(const vector<VectorXf> & features,
                   const vector<VectorXf> & labels,
                   const vector<unsigned int> & indices,
                   BTDTRNode * node)
{
    assert(node);
    const int min_leaf_node = tree_param_.min_leaf_node_;
    const int max_depth     = tree_param_.max_tree_depth_;
    const int depth = node->depth_;
    const int dim = (int)features[0].size();
    const int candidate_dim_num = tree_param_.candidate_dim_num_;
    const double min_split_stddev = tree_param_.min_split_node_std_dev_;
    assert(candidate_dim_num <= dim);
    
    // leaf node
    bool reach_leaf = false;
    if (indices.size() < min_leaf_node || depth > max_depth) {
        reach_leaf = true;
    }
    
    // check standard deviation
    if (reach_leaf == false && depth > max_depth/2) {
        Eigen::VectorXf mean;
        Eigen::VectorXf std_dev;
        DTUtil::meanStddev<Eigen::VectorXf>(labels, indices, mean, std_dev);
        // standard deviation in every dimension is smaller than the threshold
        reach_leaf = (std_dev.array() < min_split_stddev).all();
    }    
    
    // satisfy leaf node
    if (reach_leaf) {
        this->setLeafNode(features, labels, indices, node);
        return true;
    }
    
    // randomly select a subset of dimensions
    assert(dims_.size() == dim);
    std::random_shuffle(dims_.begin(), dims_.end());
    vector<unsigned int> random_dim(dims_.begin(), dims_.begin() + candidate_dim_num);
    assert(random_dim.size() > 0 && random_dim.size() <= dims_.size());
    
    // split the data to left and right node
    vector<unsigned int> left_indices;
    vector<unsigned int> right_indices;
    BTDTRSplitParameter split_param;
    bool is_split = false;
    double loss = std::numeric_limits<double>::max();
    
    
    // optimize random feature
    for (int i = 0; i<random_dim.size(); i++) {
        BTDTRSplitParameter cur_split_param;
        cur_split_param.split_dim_ = random_dim[i];
        
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        
        bool cur_is_split = bestSplitDimension(features, labels, indices, tree_param_, depth,
                                               cur_split_param,
                                               cur_left_indices,
                                               cur_right_indices);
        if (cur_is_split && cur_split_param.split_loss_ < loss) {
            is_split = true;
            loss = cur_split_param.split_loss_;
            split_param = cur_split_param;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
        }
    }
    
    // split data
    if (is_split) {
        assert(left_indices.size() + right_indices.size() == indices.size());
        if (tree_param_.verbose_) {
            printf("left percentage is %f \n", 1.0 * left_indices.size()/indices.size());
            printf("split      loss is %f \n", split_param.split_loss_);
        }
        node->split_param_ = split_param;
        node->sample_num_ = (int)indices.size();
        node->is_leaf_ = false;
        if (left_indices.size() > 0) {
            BTDTRNode *left_node = new BTDTRNode(depth + 1);
            this->configureNode(features, labels, left_indices, left_node);
            left_node->sample_percentage_ = 1.0 * left_indices.size()/indices.size();
            node->left_child_ = left_node;
        }
        if (right_indices.size() > 0) {
            BTDTRNode * right_node = new BTDTRNode(depth + 1);
            this->configureNode(features, labels, right_indices, right_node);
            right_node->sample_percentage_ = 1.0 * right_indices.size()/indices.size();
            node->right_child_ = right_node;
        }
    }
    else
    {
        this->setLeafNode(features, labels, indices, node);
        return true;
    }
    return true;
}

void BTDTRTree::setLeafNode(const vector<VectorXf> & features,
                            const vector<VectorXf> & labels,
                            const vector<unsigned int> & indices,
                            BTDTRNode * node)
{
    assert(node);
    
    node->is_leaf_ = true;
    DTUtil::meanStddev<Eigen::VectorXf>(labels, indices, node->label_mean_, node->label_stddev_);
    node->sample_num_ = (int)indices.size();
    node->feat_mean_ = DTUtil::mean(features, indices);
    leaf_node_num_++;
    
    if (tree_param_.verbose_leaf_) {
        printf("leaf node depth size %d    %lu\n", node->depth_, indices.size());
        cout<<"mean  : \n"<<node->label_mean_.transpose()<<endl;
        cout<<"stddev: \n"<<node->label_stddev_.transpose()<<endl;
    }
}




bool BTDTRTree::predict(const Eigen::VectorXf & feature,
                        const int maxCheck,
                        Eigen::VectorXf & pred) const
{
    assert(root_);    
    
    int checkCount = 0;
    float epsError = 1.0;
    const int knn = 1;
    
    BranchSt branch;
    flann::Heap<BranchSt> * heap = new flann::Heap<BranchSt>(leaf_node_num_);  // why use so large heap
    flann::DynamicBitset checked(leaf_node_num_);
 
    flann::KNNResultSet2<DistanceType> result(knn); // only keep the nearest one
    const ElementType *vec = feature.data();
    
    // search tree down to leaf
    this->searchLevel(result, vec, root_, 0, checkCount, maxCheck, epsError, heap, checked);
    
    while (heap->popMin(branch) &&
           (checkCount < maxCheck || !result.full())) {
        assert(branch.node);
        this->searchLevel(result, vec, branch.node, branch.mindist, checkCount, maxCheck, epsError, heap, checked);
    }
    
    delete heap;
    assert(result.size() == knn);
 
    size_t index = 0;
    DistanceType dist = 0.0;
    result.copy(&index, &dist, 1, false);
    
    pred = leaf_nodes_[index]->label_mean_;
    return true;
}

bool BTDTRTree::predict(const Eigen::VectorXf & feature,
                        const int maxCheck,
                        VectorXf & pred,
                        float & dist)
{
    assert(root_);
    
    int checkCount = 0;
    float epsError = 1.0;
    const int knn = 1;
    
    BranchSt branch;
    flann::Heap<BranchSt> * heap = new flann::Heap<BranchSt>(leaf_node_num_);  // why use so large heap
    flann::DynamicBitset checked(leaf_node_num_);
    
    flann::KNNResultSet2<DistanceType> result(knn); // only keep the nearest one
    const ElementType *vec = feature.data();
    
    // search tree down to leaf
    this->searchLevel(result, vec, root_, 0, checkCount, maxCheck, epsError, heap, checked);
    
    while (heap->popMin(branch) &&
           (checkCount < maxCheck || !result.full())) {
        assert(branch.node);
        this->searchLevel(result, vec, branch.node, branch.mindist, checkCount, maxCheck, epsError, heap, checked);
    }
    
    delete heap;
    assert(result.size() == knn);
    
    size_t index = 0;
    DistanceType distance;
    result.copy(&index, &distance, 1, false);
    
    pred = leaf_nodes_[index]->label_mean_;
    dist = (float)distance;
    return true;
}

void BTDTRTree::searchLevel(flann::ResultSet<DistanceType>  & result_set, const ElementType* vec, const NodePtr node,
                            const DistanceType min_dist, int & checkCount, const int maxCheck, const float epsError,
                            flann::Heap<BranchSt>* heap, flann::DynamicBitset& checked) const
{
    if (result_set.worstDist() < min_dist) {
        return;
    }
    
    // check leaf node
    if (node->is_leaf_) {
        int index = node->index_;
        if (checked.test(index) ||
            (checkCount >= maxCheck && result_set.full())) {
            return;
        }
        checked.set(index);
        checkCount++;
        
        // squared distance
        DistanceType dist = distance_(node->feat_mean_.data(), vec, node->feat_mean_.size());
        result_set.addPoint(dist, index);
        return;
    }
    
    // create a branch record for the branch not taken
    ElementType val = vec[node->split_param_.split_dim_];
    DistanceType diff = val - node->split_param_.split_threshold_;
    NodePtr bestChild  = (diff < 0 ) ? node->left_child_: node->right_child_;
    NodePtr otherChild = (diff < 0 ) ? node->right_child_: node->left_child_;
    
    DistanceType new_dist_sq = min_dist + distance_.accum_dist(val, node->split_param_.split_threshold_, node->split_param_.split_dim_);
    
    if ((new_dist_sq * epsError < result_set.worstDist()) ||
        !result_set.full()) {
        heap->insert(BranchSt(otherChild, new_dist_sq));
    }
    
    // call recursively to search next level
    this->searchLevel(result_set, vec, bestChild, min_dist, checkCount, maxCheck, epsError, heap, checked);
}

void BTDTRTree::recordLeafNodes(NodePtr node, vector<NodePtr> & leafNodes, int & index)
{
    assert(node);    
    if (node->is_leaf_) {
        // for tree read from a file, index is precomputed
        if (node->index_ != -1) {
            assert(node->index_ == index);
        }
        node->index_ = index;
        leafNodes[index] = node;
        index++;
        return;
    }
    if (node->left_child_) {
        this->recordLeafNodes(node->left_child_, leafNodes, index);
    }
    if (node->right_child_) {
        this->recordLeafNodes(node->right_child_, leafNodes, index);
    }
}

void BTDTRTree::hashLeafNode()
{
    assert(leaf_node_num_ > 0);
    leaf_nodes_.resize(leaf_node_num_);
    
    int index = 0;
    this->recordLeafNodes(root_, leaf_nodes_, index);    
    printf("tree leaf node number is %d\n", leaf_node_num_);
}

void BTDTRTree::getLeafNodeDescriptor(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(root_);
    assert(leaf_node_num_ > 0);
    assert(leaf_node_num_ == leaf_nodes_.size());
    
    const int rows = leaf_node_num_;
    const int cols = (int)leaf_nodes_[0]->feat_mean_.size();
    
    data = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(rows, cols);
    for (int i = 0; i<leaf_nodes_.size(); i++) {
        data.row(i) = leaf_nodes_[i]->feat_mean_;
    }
}

void BTDTRTree::setLeafNodeDescriptor(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(root_);
    assert(leaf_node_num_ > 0);
    assert(leaf_node_num_ == data.rows());
    
    this->hashLeafNode();
    for (int i = 0; i<leaf_nodes_.size(); i++) {
        leaf_nodes_[i]->feat_mean_ = data.row(i);
    }
}

const BTDTRTreeParameter & BTDTRTree::getTreeParameter(void) const
{
    return tree_param_;
}
void BTDTRTree::setTreeParameter(const BTDTRTreeParameter & param)
{
    tree_param_ = param;    
}









