//
//  btdtr_ptz_util.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-07-20.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "btdtr_ptz_util.h"
#include "mat_io.hpp"
#include "eigen_geometry_util.h"

namespace btdtr_ptz_util {
PTZTreeParameter::PTZTreeParameter()
{
    sampled_frame_num_ = 50;
    pp_x_ = 1280.0/2;
    pp_y_ = 720.0/2;
}

    PTZTreeParameter::PTZTreeParameter(const PTZTreeParameter& other) {
    if (this == &other) {
        return;
    }
    sampled_frame_num_ = other.sampled_frame_num_;
    base_tree_param_ = other.base_tree_param_;
}

bool  PTZTreeParameter::readFromFile(FILE *pf)
{
    assert(pf);
    
    const int param_num = 3;
    unordered_map<std::string, int> imap;
    for(int i = 0; i<param_num; i++)
    {
        char s[1024] = {NULL};
        int val = 0;
        int ret = fscanf(pf, "%s %d", s, &val);
        if (ret != 2) {
            break;
        }
        imap[string(s)] = val;
    }
    assert(imap.size() == 3);
    
    sampled_frame_num_ = imap[string("sampled_frame_num")];
    pp_x_ = imap[string("pp_x")];
    pp_y_ = imap[string("pp_y")];
    base_tree_param_.readFromFile(pf);
    return true;
}

bool  PTZTreeParameter::readFromFile(const char *file_name)
{
    assert(file_name);
    FILE *pf = fopen(file_name, "r");
    if(!pf){
        printf("can not open %s\n", file_name);
        return false;
    }
    this->readFromFile(pf);
    fclose(pf);
    return true;
}

bool PTZTreeParameter::writeToFile(FILE *pf) const
{
    assert(pf);
    fprintf(pf, "sampled_frame_num %d\n", sampled_frame_num_);
    fprintf(pf, "pp_x %f\n", pp_x_);
    fprintf(pf, "pp_y %f\n\n", pp_y_);
    base_tree_param_.writeToFile(pf);
    return true;
}

void PTZTreeParameter::printSelf() const
{
    writeToFile(stdout);
}

// read sample location and sift feature
static bool readSIFTLocationAndDescriptors(const char *mat_file,
            vector<Eigen::VectorXf> & locations,
            vector<Eigen::VectorXf> & features)
{
    //bool readMatrix(const char *file_name, const char *var_name, matrixT & data);
    Eigen::MatrixXf keypoint;
    Eigen::MatrixXf descriptor;
    bool is_read = matio::readMatrix(mat_file, "keypoint", keypoint, false);
    assert(is_read);
    is_read = matio::readMatrix(mat_file, "descriptor", descriptor, false);
    assert(is_read);
    assert(keypoint.cols() == descriptor.cols());

    const int dims = (int) descriptor.rows();
    Eigen::VectorXf loc = VectorXf::Zero(2, 1);
    Eigen::VectorXf feat = VectorXf::Zero(dims, 1);
    for (int i = 0; i<keypoint.cols(); i++) {
        loc[0] = keypoint(0, i);
        loc[1] = keypoint(1, i);
        feat = descriptor.col(i);

        locations.push_back(loc);
        features.push_back(feat);
    }
    assert(locations.size() == features.size());
    //printf("load %lu features from %s\n", features.size(), mat_file);
    return true;
}
    static bool readPTZFeatureLocationAndDescriptors(const char *mat_file,
                                                     Eigen::Vector3f & ptz,
                                                     vector<Eigen::VectorXf> & locations,
                                                     vector<Eigen::VectorXf> & descriptors)
    {
        
        Eigen::MatrixXf keypoint_data;
        Eigen::MatrixXf descriptor_data;
        Eigen::MatrixXf ptz_data;
        bool is_read = matio::readMatrix(mat_file, "keypoint", keypoint_data, false);
        assert(is_read);
        is_read = matio::readMatrix(mat_file, "descriptor", descriptor_data, false);
        assert(is_read);
        assert(keypoint_data.rows() == descriptor_data.rows());
        
        is_read = matio::readMatrix(mat_file, "ptz", ptz_data, false);
        assert(ptz_data.rows() == 3 && ptz_data.cols() == 1);
        ptz[0] = ptz_data(0, 0);
        ptz[1] = ptz_data(1, 0);
        ptz[2] = ptz_data(2, 0);
        
        const int dims = (int) descriptor_data.cols();
        Eigen::VectorXf loc = VectorXf::Zero(2, 1);
        Eigen::VectorXf feat = VectorXf::Zero(dims, 1);
        for (int i = 0; i<keypoint_data.rows(); i++) {
            loc[0] = keypoint_data(i, 0);
            loc[1] = keypoint_data(i, 1);
            feat = descriptor_data.row(i);
            
            locations.push_back(loc);
            descriptors.push_back(feat);
        }
        assert(locations.size() == descriptors.size());
        
        return true;
    }
    
void generatePTZSample(const char* feature_file_name,
                       const Eigen::Vector2f& pp,
                       const Eigen::Vector3f& ptz,
                       vector<PTZSample>& samples)
{
    
    assert(feature_file_name);

    vector<Eigen::VectorXf> locations;
    vector<Eigen::VectorXf> features;
    readSIFTLocationAndDescriptors(feature_file_name, locations, features);        
    for (int i = 0; i<locations.size(); i++) {
        PTZSample s;
        Eigen::Vector2f pan_tilt;
        s.loc_[0] = locations[i][0];
        s.loc_[1] = locations[i][1];       
        EigenX::pointPanTilt(pp, ptz, s.loc_, pan_tilt);
        s.pan_tilt_[0] = pan_tilt[0];
        s.pan_tilt_[1] = pan_tilt[1];
        s.descriptor_ = features[i];
        samples.push_back(s);
    }
}
    
    void generatePTZSampleWithFeature(const char * feature_ptz_file_name,
                                      const Eigen::Vector2f& pp,
                                      Eigen::Vector3f & ptz,
                                      vector<PTZSample> & samples)
    {
        assert(feature_ptz_file_name);
        
        vector<Eigen::VectorXf> locations;
        vector<Eigen::VectorXf> features;
        readPTZFeatureLocationAndDescriptors(feature_ptz_file_name, ptz, locations, features);
        for (int i = 0; i<locations.size(); i++) {
            PTZSample s;
            Eigen::Vector2f pan_tilt;
            s.loc_[0] = locations[i][0];
            s.loc_[1] = locations[i][1];
            EigenX::pointPanTilt(pp, ptz, s.loc_, pan_tilt);
            s.pan_tilt_[0] = pan_tilt[0];
            s.pan_tilt_[1] = pan_tilt[1];
            s.descriptor_ = features[i];
            samples.push_back(s);
        }
    }
    
void readSequenceData(const char * sequence_file_name,
                                        const char * sequence_base_directory,
                                        vector<string> & feature_files,
                                        vector<Eigen::Vector3f> & ptzs)
{
    assert(sequence_file_name);
    assert(sequence_base_directory);
    
    FILE *pf = fopen(sequence_file_name, "r");
    if (!pf) {
        printf("can not open: %s\n", sequence_file_name);
        return;
    }
    // skip three rows;
    for (int i = 0; i<3; i++) {
        char buf[1024] = {NULL};
        fgets(buf, sizeof(buf), pf);
        printf("%s", buf);
    }
    while (1) {
        char buf[1024] = {NULL};
        Eigen::Vector3f ptz;
        double pan = 0, tilt = 0, fl = 0;
        int ret = fscanf(pf, "%s %lf %lf %lf", buf, &pan, &tilt, &fl);
        if (ret != 4) {
            break;
        }
        ptz[0] = pan;
        ptz[1] = tilt;
        ptz[2] = fl;
        feature_files.push_back(sequence_base_directory + string(buf));
        ptzs.push_back(ptz);
    }
    assert(ptzs.size() == feature_files.size());
    fclose(pf);
    printf("load %lu files\n", feature_files.size());
}



};