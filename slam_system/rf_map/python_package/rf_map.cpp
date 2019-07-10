//
//  rf_map.cpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-07-03.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include <string>
#include <vector>
#include <fstream>
#include "rf_map.hpp"
#include "rf_map_builder.hpp"
#include "btdtr_ptz_util.h"
#include "ptz_pose_estimation.h"

using namespace std;

RFMap::RFMap()
{
    
}

RFMap::~RFMap()
{
    
}
// Create a model from a list of feature_label files
void RFMap::createMap(const char * feature_label_file,
                        const char * model_parameter_file,
                        const char * model_name)
{
    // 1. read feature label file
    vector<string> feature_files;
    ifstream file(feature_label_file);
    string str;
    while(std::getline(file, str)) {
        feature_files.push_back(str);
    }
    printf("read %lu feature label files\n", feature_files.size());
    
    btdtr_ptz_util::PTZTreeParameter tree_param;
    tree_param.readFromFile(model_parameter_file);
    //tree_param.printSelf();
    
    RFMapBuilder builder;
    builder.setTreeParameter(tree_param);
    
    //printf("start build model\n");
    
    builder.buildModel(model_, feature_files, model_name, false);
    model_.saveModel(model_name);
    printf("save model to file %s\n", model_name);
}

// relocalize a camera using the model
// parameter_file: testing parameter
// pan_tilt_zoom: output
void RFMap::relocalizeCamera(const char* feature_location_file_name,
                            const char* test_parameter_file,
                            double* pan_tilt_zoom)
{
    int max_check = 4;
    double distance_threshold = 0.2;
    Eigen::Vector2f pp(1280/2.0, 720/2.0);
    ptz_pose_opt::PTZPreemptiveRANSACParameter ransac_param;
    ransac_param.reprojection_error_threshold_ = 2.0;
    ransac_param.sample_number_ = 32;
    
    vector<btdtr_ptz_util::PTZSample> samples;
    btdtr_ptz_util::generatePTZSampleWithFeature(feature_location_file_name,
                                                 pp, samples);
    printf("feature number is %lu\n", samples.size());
    
    vector<Eigen::Vector2d> image_points;
    vector<vector<Eigen::Vector2d> > candidate_pan_tilt;
    Eigen::Vector3d estimated_ptz(pan_tilt_zoom[0], pan_tilt_zoom[1], pan_tilt_zoom[2]);
    // predict from observation (descriptors)
    double tt = clock();
    for (int j = 0; j<samples.size(); j++) {
        btdtr_ptz_util::PTZSample s = samples[j];
        Eigen::VectorXf feat = s.descriptor_;
        vector<Eigen::VectorXf> cur_predictions;
        vector<float> cur_dists;
        model_.predict(feat, max_check, cur_predictions, cur_dists);
        assert(cur_predictions.size() == cur_dists.size());
        
        //cout<<"minimum feature distance "<<cur_dists[0]<<endl;
        if (cur_dists[0] < distance_threshold) {
            image_points.push_back(Eigen::Vector2d(s.loc_.x(), s.loc_.y()));
            vector<Eigen::Vector2d> cur_candidate;
            for (int k = 0; k<cur_predictions.size(); k++) {
                assert(cur_predictions[k].size() == 2);
                if (cur_dists[k] < distance_threshold) {
                    cur_candidate.push_back(Eigen::Vector2d(cur_predictions[k][0], cur_predictions[k][1]));
                }
            }
            //printf("min feature distance is %lf\n", cur_dists[0]);
            candidate_pan_tilt.push_back(cur_candidate);
        }
    }
    printf("candidate point number %lu\n", candidate_pan_tilt.size());
    // estimate camera pose
    bool is_opt = ptz_pose_opt::preemptiveRANSACOneToMany(image_points, candidate_pan_tilt,
                                                          pp.cast<double>(),
                                                          ransac_param, estimated_ptz, false);
    printf("Prediction and camera pose estimation cost time: %f seconds.\n", (clock() - tt)/CLOCKS_PER_SEC);
    if (!is_opt) {
        printf("-------------------------------------------- Optimize PTZ failed.\n");
        printf("valid feature number is %lu\n\n", image_points.size());
    }
    else {
        pan_tilt_zoom[0] = estimated_ptz[0];
        pan_tilt_zoom[1] = estimated_ptz[1];
        pan_tilt_zoom[2] = estimated_ptz[2];
    }
}

void RFMap::estimateCameraRANSAC(const char* pixel_ray_file_name,
                               double* pan_tilt_zoom)
{
    // RANSAC parameter
    Eigen::Vector2d pp(1280/2.0, 720/2.0);
    ptz_pose_opt::PTZPreemptiveRANSACParameter ransac_param;
    ransac_param.reprojection_error_threshold_ = 2.0;
    ransac_param.sample_number_ = 32;
    
    // read pixel-ray correspondences
    vector<Eigen::Vector2d> image_points;
    vector<Eigen::Vector2d> rays;
    btdtr_ptz_util::readKeypointRay(pixel_ray_file_name, image_points, rays);
    
    vector<vector<Eigen::Vector2d> > candidate_pan_tilt;
    for (int i = 0; i<image_points.size(); i++) {
        vector<Eigen::Vector2d> pan_tilt;
        pan_tilt.push_back(rays[i]);
        candidate_pan_tilt.push_back(pan_tilt);
    }
    
    Eigen::Vector3d estimated_ptz(pan_tilt_zoom[0], pan_tilt_zoom[1], pan_tilt_zoom[2]);
    
    // estimate camera pose
    double tt = clock();
    bool is_opt = ptz_pose_opt::preemptiveRANSACOneToMany(image_points, candidate_pan_tilt,
                                                          pp,
                                                          ransac_param, estimated_ptz, false);
    printf("Prediction and camera pose estimation cost time: %f seconds.\n", (clock() - tt)/CLOCKS_PER_SEC);
    if (!is_opt) {
        printf("-------------------------------------------- Optimize PTZ failed.\n");
    }
    else {
        pan_tilt_zoom[0] = estimated_ptz[0];
        pan_tilt_zoom[1] = estimated_ptz[1];
        pan_tilt_zoom[2] = estimated_ptz[2];
    }
}

/******************-----------C inter face--------------******************/
EXPORTIT RFMap* RFMap_new()
{
    RFMap* p_map = new RFMap();
    //printf("Debug: before address %p\n", (void*)p_map);
    //printf("Before: value %lld\n", (long long)p_map);
    return p_map;
}

EXPORTIT void RFMap_delete(RFMap* rf_map)
{
    delete rf_map;
    rf_map = nullptr;
}

EXPORTIT void relocalizeCamera(RFMap* rf_map,
                              const char* feature_location_file_name,
                              const char* test_parameter_file,
                              double* pan_tilt_zoom)
{
    //printf("Debug: after 2 address %p\n", (void*)rf_map);
    rf_map->relocalizeCamera(feature_location_file_name, test_parameter_file, pan_tilt_zoom);
}

EXPORTIT void createMap(RFMap* rf_map,
                        const char * feature_label_file,
                        const char * model_parameter_file,
                        const char * model_name)
{
    //printf("after 1: address %p\n", (void*)rf_map);
    //printf("after: address %p\n", (void*)(&(rf_map->model_)));
    rf_map->createMap(feature_label_file, model_parameter_file, model_name);
    //printf("after 1 1: address %p\n", (void*)rf_map);
}

EXPORTIT void estimateCameraRANSAC(const char* pixel_ray_file_name,
                                   double* pan_tilt_zoom)
{
    RFMap::estimateCameraRANSAC(pixel_ray_file_name, pan_tilt_zoom);
}
