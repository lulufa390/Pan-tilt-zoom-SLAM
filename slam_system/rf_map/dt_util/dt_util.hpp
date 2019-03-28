//
//  dt_util.h
//  Classifer_RF
//
//  Created by jimmy on 2017-02-16.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__dt_util__
#define __Classifer_RF__dt_util__

// decision tree util
#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include <unordered_map>
#include <string>

using std::vector;
using std::string;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::VectorXi;

using std::vector;

namespace dt {
    // randomly generate a subset of dimensions
    template<class intType>
    vector<intType> randomDimension(const intType dim, const intType num);
    
    template <class intType>
    vector<intType> range(int start, int end, int step)
    {
        assert((end - start) * step >= 0);
        vector<intType> ret;
        for (int i = start; i < end; i += step) {
            ret.push_back((intType)i);
        }
        return ret;
    }
    
    // mean and standard deviation
    template <class vectorType>
    void meanStd(const vector<vectorType> & labels, vectorType & mean, vectorType & sigma);
    
    template<class vectorType, class intType>
    void meanStd(const vector<vectorType> & labels, const vector<intType> & indices,
                 vectorType & mean, vectorType & sigma);
    
    // balance examples in each category
    // return: example indices with balanced training examples
    template<class intType>
    vector<intType> balanceSamples(const vector<intType> & example_indices,
                                   const vector<intType> & labels, const int category_num);
    
    // regression loss
    template<class VectorType, class IntType>
    double sumOfVariance(const vector<VectorType> & labels, const vector<IntType> & indices);
    
    // find most common number in the vector
    template<class intType>
    intType mostCommon(const vector<intType> & data);
    
    template <class VectorType>
    void meanMedianError(const vector<VectorType> & errors, VectorType & mean, VectorType & median);
    
}  // namespace

class DTUtil
{
public:
    // randomly generate a subset of dimensions
    static vector<unsigned int> randomDimensions(const int dimension, const int ccandidate_dimension);
    
    template <class T>
    static double spatialVariance(const vector<T> & labels, const vector<unsigned int> & indices);
    
    // full variance of Gaussian model
    template <class T>
    static double fullVariance(const vector<T>& labels, const vector<unsigned int> & indices);
    
    template <class MatrixType>
    static double sumOfVariance(const vector<MatrixType> & labels, const int row_index,
                                const vector<unsigned int> & indices);
    
    template <class Type1, class Type2>
    static double spatialVariance(const vector<Type1> & labels,
                                  const vector<unsigned int> & indices, const vector<Type2> & wt);
    
    template <class T>
    static void meanStddev(const vector<T> & labels, const vector<unsigned int> & indices, T & mean, T & sigma);
    
    template <class vectorT, class indexT>
    static vectorT mean(const vector<vectorT> & data, const vector<indexT> & indices);
    
    template <class T>
    static T mean(const vector<T> & data);
    
    // mean and standard of particular row
    template <class matrixType, class vectorType>
    static void rowMeanStddev(const vector<matrixType> & labels, const vector<unsigned int> & indices,
                              const int row_index, vectorType & mean,   vectorType & sigma);
    
   
    
    // https://en.wikipedia.org/wiki/Quartile
    // q1, q2, q3: first, second and third quartile. The second quartile is median
    template <class vectorT>
    static void quartileError(const vector<vectorT> & errors, vectorT & q1, vectorT& q2, vectorT& q3);
    
    // mean error of each row of a list of matrixes
    template <class MatrixType>
    static void matrixMeanError(const vector<MatrixType> & errors, MatrixType & mean);
    
   
    static double crossEntropy(const Eigen::VectorXd& prob);
    
    static double crossEntropy(const Eigen::VectorXf& prob);   
    
    
    static double balanceLoss(const int leftNodeSize, const int rightNodeSize);
    
    static bool isSameLabel(const vector<unsigned int> & labels, const vector<unsigned int> & indices);
    static bool isSameLabel(const vector<int>& labels, const vector<int>& indices);
    
    // minimum number of examples in all category
    static int minLabelNumber(const vector<unsigned int> & labels,
                              const vector<unsigned int> & indices,
                              const int num_category);
    
    // label is a sequential data
    static int minLabelNumber(const vector<VectorXi> & labels,
                              const vector<unsigned int> & indices,
                              const int time_step,
                              const int num_category);
    
    template <class integerType>
    static Eigen::MatrixXd confusionMatrix(const vector<integerType> & predictions,
                                           const vector<integerType> & labels,
                                           const int category_num,
                                           bool normalize);
    
    // accuracy (should be precision) of each category and average
    static Eigen::VectorXd accuracyFromConfusionMatrix(const Eigen::MatrixXd & conf);
    static Eigen::VectorXd precisionFromConfusionMatrix(const Eigen::MatrixXd & conf);
    
    template <class T>
    static vector<T> range(int start, int end, int step)
    {
        assert((end - start) * step >= 0);
        vector<T> ret;
        for (int i = start; i < end; i += step) {
            ret.push_back((T)i);
        }
        return ret;
    }

    
};

#endif /* defined(__Classifer_RF__dt_util__) */
