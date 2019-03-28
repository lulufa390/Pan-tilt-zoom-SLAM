//
//  DTRandom.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-09-20.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "dt_random.hpp"
#include "vnl_random.h"
#include <cassert>


DTRandom::DTRandom()
{
    rnd_generator_ = new vnl_random();
}

DTRandom::~DTRandom()
{
    if (rnd_generator_) {
        delete rnd_generator_;
        rnd_generator_ = NULL;
    }
}

double DTRandom::getRandomNumber(const double min_v, const double max_v) const
{
    assert(rnd_generator_);
    return rnd_generator_->drand32(min_v, max_v);
}

vector<double> DTRandom::getRandomNumbers(const double min_v, const double max_v, int num) const
{
    assert(rnd_generator_);
    
    assert(min_v < max_v);
    
    vector<double> values;
    for (int i = 0; i<num; i++) {
        double v = rnd_generator_->drand32(min_v, max_v);
        values.push_back(v);
    }
    return values;
}

template <class integerT>
void DTRandom::outofBagSample(const unsigned int N,
                    vector<integerT> & bootstrapped,
                    vector<integerT> & outof_bag)
{
    vector<bool> isPicked(N, false);
    for (int i = 0; i<N; i++) {
        integerT idx = rnd_generator_->lrand32(0, N-1);
        bootstrapped.push_back(idx);
        isPicked[idx] = true;
    }
    
    for (int i = 0; i<N; i++) {
        if (!isPicked[i]) {
            outof_bag.push_back(i);
        }
    }
}

template <class integerT>
void DTRandom::outofBagSampling(const unsigned int N,
                                  vector<integerT> & bootstrapped,
                                  vector<integerT> & outof_bag)
{
    vnl_random rnd;
    
    vector<bool> isPicked(N, false);
    for (int i = 0; i<N; i++) {
        integerT idx = rnd.lrand32(0, N-1);
        bootstrapped.push_back(idx);
        isPicked[idx] = true;
    }
    
    for (int i = 0; i<N; i++) {
        if (!isPicked[i]) {
            outof_bag.push_back(i);
        }
    }
}

vector<double>
DTRandom::generateRandomNumber(const double min_v, const double max_v, int num)
{
    assert(min_v < max_v);
    
    vector<double> values;
    vnl_random rnd;
    for (int i = 0; i<num; i++) {
        double v = rnd.drand32(min_v, max_v);
        values.push_back(v);
    }
    return values;
}

/*
double
DTRandom::randomNumber(const double min_v, const double max_v)
{
    assert(min_v < max_v);
    vnl_random rnd;
    return rnd.drand32(min_v, max_v);
}
 */

template
void DTRandom::outofBagSample(const unsigned int N,
                              vector<int> & bootstrapped,
                              vector<int> & outof_bag);

template
void DTRandom::outofBagSampling(const unsigned int N,
                                vector<int> & bootstrapped,
                                vector<int> & outof_bag);

template
void DTRandom::outofBagSampling(const unsigned int N,
                                vector<unsigned int> & bootstrapped,
                                vector<unsigned int> & outof_bag);
