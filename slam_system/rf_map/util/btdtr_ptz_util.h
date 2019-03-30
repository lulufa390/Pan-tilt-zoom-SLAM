//
//  btdtr_ptz_util.h
//  PTZBTRF
//
//  Created by jimmy on 2017-07-20.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __PTZBTRF__btdtr_ptz_util__
#define __PTZBTRF__btdtr_ptz_util__

// backtracking decision tree regressor
// soccer field relocalization
#include <stdio.h>
#include <Eigen/Dense>
#include <unordered_map>
#include <string>
#include "bt_dtr_util.h"

using std::unordered_map;
using std::string;

namespace btdtr_ptz_util {

class PTZSample
{
public:
    Eigen::Vector2f  loc_;     // 2D location (x, y)
    Eigen::VectorXf  pan_tilt_;       //  Pan, tilt parameter in world coordinate, label
    Eigen::VectorXf  descriptor_; // image patch descriptor, feature
    
    PTZSample() {
        pan_tilt_ = Eigen::VectorXf::Zero(2, 1);
    }
    
};

class PTZTreeParameter
{
public:
    int sampled_frame_num_;           // sampled frames for a tree
    double pp_x_;                     // principal point x
    double pp_y_;
    
    BTDTRTreeParameter base_tree_param_;   // general tacktracking regression tree parameter

    PTZTreeParameter();
    PTZTreeParameter(const PTZTreeParameter& other);
    
    bool readFromFile(FILE *pf);
    bool readFromFile(const char *file_name);
    bool writeToFile(FILE *pf) const;
    void printSelf() const;
};
    
//
    // pp: principal point
    // ptz: pan, tilt and focal length of the image
    // samples: sample data, has No feature descriptor
void generatePTZSample(const char* feature_file_name,
                       const Eigen::Vector2f& pp,
                       const Eigen::Vector3f& ptz,
                       vector<PTZSample>& samples);
    
    //feature_ptz_file_name: .mat file
    // contains: im_name (str), camera (9x1), ptz (3x1), keypoint (nx2), descriptor (nx128)
void generatePTZSampleWithFeature(const char * feature_ptz_file_name,
                                  const Eigen::Vector2f& pp,
                                  Eigen::Vector3f & ptz,
                                  vector<PTZSample> & samples);

void readSequenceData(const char * sequence_file_name,
                      const char * sequence_base_directory,
                        vector<string> & feature_files,
                        vector<Eigen::Vector3f> & ptzs);
    

    
    
}; // namespace btdtr_ptz_util


#endif /* defined(__PTZBTRF__btdtr_ptz_util__) */
