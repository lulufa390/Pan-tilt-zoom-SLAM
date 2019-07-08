//
//  online_rf_map.cpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-07-06.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "online_rf_map.hpp"
#include "ptz_pose_estimation.h"

OnlineRFMap::OnlineRFMap()
{
    
}

OnlineRFMap::~OnlineRFMap()
{
    
}

// create a map from a single feature label file
void OnlineRFMap::createMap(const char * feature_label_file,
                          const char * model_parameter_file,
                          const char * model_name)
{
    btdtr_ptz_util::PTZTreeParameter tree_param;
    tree_param.readFromFile(model_parameter_file);
    
    builder_.setTreeParameter(tree_param);
    builder_.addTree(model_, feature_label_file, model_name, false);
}

// update a map: may add or update a tree
void OnlineRFMap::updateMap(const char * feature_label_file,
                          const char * model_name)
{
    const double error_threshold = 0.1;
    const double percentage_threshold = 0.5;
    bool is_add = builder_.isAddTree(model_, string(feature_label_file),
                                     error_threshold, percentage_threshold);
    if (is_add) {
        builder_.addTree(model_, feature_label_file, model_name, false);
    }
    else {
        builder_.updateTree(model_, feature_label_file, model_name, false);
    }
}



void OnlineRFMap::relocalizeCamera(const char* feature_location_file_name,
                                 const char* test_parameter_file,
                                 double* pan_tilt_zoom)
{
    //@todo the same code as in RFMap
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


/******************-----------C inter face--------------******************/
EXPORTIT OnlineRFMap* OnlineRFMap_new()
{
    OnlineRFMap* ol_rf_map = new OnlineRFMap();
    return ol_rf_map;
}

EXPORTIT void OnlineRFMap_delete(OnlineRFMap* ol_rf_map)
{
    assert(ol_rf_map != nullptr);
    delete ol_rf_map;
    ol_rf_map = nullptr;
}

EXPORTIT void createOnlineMap(OnlineRFMap* ol_rf_map,
                              const char * feature_label_file,
                              const char * model_parameter_file,
                              const char * model_name)
{
    assert(ol_rf_map != nullptr);
    ol_rf_map->createMap(feature_label_file, model_parameter_file, model_name);
}

EXPORTIT void updateOnlineMap(OnlineRFMap* ol_rf_map,
                              const char * feature_label_file,
                              const char * model_name)
{
    assert(ol_rf_map != nullptr);
    ol_rf_map->updateMap(feature_label_file, model_name);
}

EXPORTIT void relocalizeCameraOnline(OnlineRFMap* ol_rf_map,
                               const char* feature_location_file_name,
                               const char* test_parameter_file,
                               double* pan_tilt_zoom)
{
    assert(ol_rf_map != nullptr);
    ol_rf_map->relocalizeCamera(feature_location_file_name, test_parameter_file, pan_tilt_zoom);
}
