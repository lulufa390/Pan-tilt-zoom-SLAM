//
//  BTDTRUtil.cpp
//  RGBD_RF
//
//  Created by jimmy on 2016-12-29.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "bt_dtr_util.h"

/*
template <class T>
double BTDTRUtil::spatialVariance(const vector<T> & labels, const vector<unsigned int> & indices)
{
    if (indices.size() <= 0) {
        return 0.0;
    }
    assert(indices.size() > 0);
    
    T mean = T::Zero(labels[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index];
    }
    mean /= indices.size();
    
    double var = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        T dif = labels[index] - mean;
        for (int j = 0; j<dif.size(); j++) {
            var += dif[j] * dif[j];
        }
    }
    return var;
}

template <class T>
void BTDTRUtil::meanStddev(const vector<T> & labels, const vector<unsigned int> & indices, T & mean, T & sigma)
{
    assert(indices.size() > 0);
    
    mean = T::Zero(labels[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < labels.size());
        mean += labels[index];
    }
    mean /= indices.size();
    
    sigma = T::Zero(labels[0].size());
    if (indices.size() == 1) {
        return;
    }
    for (int i = 0; i<indices.size(); i++) {
        T dif = labels[indices[i]] - mean;
        for (int j = 0; j<sigma.size(); j++) {
            sigma[j] += dif[j] * dif[j];
        }
    }
    for (int j = 0; j<sigma.size(); j++) {
        sigma[j] = sqrt(fabs(sigma[j])/indices.size());
    }
}

template <class T>
void BTDTRUtil::mean_stddev(const vector<T>& labels, T & mean, T & sigma)
{
    assert(labels.size() > 0);
    
    // mean value
    mean = T::Zero(labels[0].size());
    for (int i = 0; i<labels.size(); i++) {
        mean += labels[i];
    }
    mean /= labels.size();
    
    sigma = T::Zero(labels[0].size());
    if (labels.size() == 1) {
        return;
    }
    
    // standard deviation
    for (int i = 0; i<labels.size(); i++) {
        T dif = labels[i] - mean;
        for (int j = 0; j<sigma.size(); j++) {
            sigma[j] += dif[j] * dif[j];
        }
    }
    for (int j = 0; j<sigma.size(); j++) {
        sigma[j] = sqrt(fabs(sigma[j])/labels.size());
    }
}

template <class T>
T BTDTRUtil::mean(const vector<T> & data, const vector<unsigned int> & indices)
{
    assert(indices.size() > 0);
    
    T m = T::Zero(data[0].size());
    
    for (int i = 0; i<indices.size(); i++) {
        int index = indices[i];
        assert(index >= 0 && index < data.size());
        m += data[index];
    }
    m /= indices.size();
    
    return m;
}

template <class T>
T BTDTRUtil::mean(const vector<T> & data)
{
    assert(data.size() > 0);
    
    T m = T::Zero(data[0].size());
    for (int i = 0; i<data.size(); i++) {
        m += data[i];
    }
    m /= data.size();
    return m;
}

template <class T>
void BTDTRUtil::mean_median_error(const vector<T> & errors,
                       T & mean,
                       T & median)
{
    assert(errors.size() > 0);
    const int dim = (int)errors[0].size();
    mean = T::Zero(dim);
    median = T::Zero(dim);
    
    vector<vector<double> > each_dim_data(dim);
    for (int i = 0; i<errors.size(); i++) {
        T err = errors[i].cwiseAbs();
        mean += err;
        for (int j = 0; j<err.size(); j++) {
            each_dim_data[j].push_back(err[j]);
        }
    }
    mean /= errors.size();
    
    for (int i = 0; i<each_dim_data.size(); i++) {
        std::sort(each_dim_data[i].begin(), each_dim_data[i].end());
        median[i] = each_dim_data[i][each_dim_data[i].size()/2];
    }
}

double BTDTRUtil::inbalance_loss(const int leftNodeSize, const int rightNodeSize)
{
    double dif = leftNodeSize - rightNodeSize;
    double num = leftNodeSize + rightNodeSize;
    double loss = fabs(dif)/num;
    assert(loss >= 0);
    return loss;
}



template double BTDTRUtil::spatialVariance(const vector<Eigen::VectorXf> & labels, const vector<unsigned int> & indices);

template void BTDTRUtil::meanStddev(const vector<Eigen::VectorXf> & labels,
                                    const vector<unsigned int> & indices,
                                    Eigen::VectorXf & mean, Eigen::VectorXf & sigma);

template void BTDTRUtil::mean_stddev(const vector<Eigen::VectorXf>& labels, Eigen::VectorXf & mean, Eigen::VectorXf & sigma);

template void BTDTRUtil::mean_stddev(const vector<Eigen::Vector3f>& labels, Eigen::Vector3f & mean, Eigen::Vector3f & sigma);

template Eigen::VectorXf BTDTRUtil::mean(const vector<Eigen::VectorXf> & data, const vector<unsigned int> & indices);

template Eigen::VectorXf BTDTRUtil::mean(const vector<Eigen::VectorXf> & data);

template void  BTDTRUtil::mean_median_error(const vector<Eigen::VectorXf> & errors, Eigen::VectorXf & mean, Eigen::VectorXf & median);
 */




