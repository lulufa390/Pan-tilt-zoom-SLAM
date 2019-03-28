//  Created by jimmy on 2016-12-29.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "bt_dtr_node.h"

void BTDTRNode::writeNode(FILE *pf, const NodePtr node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    
    // write current node
    BTDTRNode::SplitParameter param = node->split_param_;
    fprintf(pf, "%2d\t %d\t %6d\t\t %lf\t %.2f\t\t %d\n",
            node->depth_, (int)node->is_leaf_,  param.split_dim_, param.split_threshold_, node->sample_percentage_, node->sample_num_);
    
    if (node->is_leaf_) {
        // leaf index and mean size
        fprintf(pf, "%d\t %d\n", node->index_, (int)node->label_mean_.size());
        for (int i = 0; i<node->label_mean_.size(); i++) {
            fprintf(pf, "%lf ", node->label_mean_[i]);
        }
        fprintf(pf, "\n");
        for (int i = 0; i<node->label_stddev_.size(); i++) {
            fprintf(pf, "%lf ", node->label_stddev_[i]);
        }
        fprintf(pf, "\n");
    }
    
    BTDTRNode::writeNode(pf, node->left_child_);
    BTDTRNode::writeNode(pf, node->right_child_);
}


bool BTDTRNode::writeTree(const char *fileName, const NodePtr root, const int leafNodeNum)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    fprintf(pf, "%d\n", leafNodeNum);
    fprintf(pf, "depth\t isLeaf\t splitDim\t threshold\t percentage\t num\t mean\t stddev\n");
    BTDTRNode::writeNode(pf, root);
    fclose(pf);
    return true;
}

bool BTDTRNode::readTree(const char *fileName, NodePtr & root, int & leafNodeNum)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not open file %s\n", fileName);
        return false;
    }
    
    // leaf node number
    int ret = fscanf(pf, "%d", &leafNodeNum);
    assert(ret == 1);
    
    // remove '\n' at the end of the line
    char dummy_line_buf[1024] = {NULL};
    fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
    
    //read marking line
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf), pf);
    printf("%s\n", line_buf);
   
    BTDTRNode::readNode(pf, root);
    fclose(pf);
    return true;
}

void BTDTRNode::readNode(FILE *pf, NodePtr & node)
{
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    if (!ret) {
        node = NULL;
        return;
    }
    if (lineBuf[0] == '#') {
        // empty node
        node = NULL;
        return;
    }
    
    // read node parameters
    node = new BTDTRNode(0);
    assert(node);
    int depth = 0;
    int is_leaf = 0;
    int split_dim = 0;
    double split_threshold = 0.0;
    int sample_num = 0;
    double sample_percentage = 0.0;
    
    int ret_num = sscanf(lineBuf, "%d %d %d %lf %lf %d",
                         &depth, &is_leaf, &split_dim, &split_threshold, &sample_percentage, &sample_num);
    assert(ret_num == 6);
    
    node->depth_ = depth;
    node->is_leaf_ = is_leaf;
    node->sample_num_ = sample_num;
    node->sample_percentage_ = sample_percentage;
    
    BTDTRNode::SplitParameter param;
    param.split_dim_ = split_dim;
    param.split_threshold_ = split_threshold;
    node->split_param_ = param;
    
    if (is_leaf) {
        int label_dim = 0;
        int index = 0;
        ret_num = fscanf(pf, "%d %d", &index, &label_dim);
        assert(ret_num == 2);
        Eigen::VectorXf mean = Eigen::VectorXf::Zero(label_dim);
        Eigen::VectorXf stddev = Eigen::VectorXf::Zero(label_dim);
        for (int i = 0; i<label_dim; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            mean[i] = val;
        }
        for (int i = 0; i<label_dim; i++) {
            double val = 0;
            ret_num = fscanf(pf, "%lf", &val);
            assert(ret_num);
            stddev[i] = val;
        }
        // remove '\n' at the end of the line
        char dummy_line_buf[1024] = {NULL};
        fgets(dummy_line_buf, sizeof(dummy_line_buf), pf);
        node->label_mean_ = mean;
        node->label_stddev_ = stddev;
        node->index_ = index;
    }
    
    BTDTRNode::readNode(pf, node->left_child_);
    BTDTRNode::readNode(pf, node->right_child_);
}

