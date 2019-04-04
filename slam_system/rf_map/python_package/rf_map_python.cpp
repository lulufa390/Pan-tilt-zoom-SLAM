//
//  rf_map_python.cpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-03-30.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "rf_map_python.hpp"
#include <string>
#include <vector>
#include <fstream>
#include "rf_map_builder.hpp"
#include "bt_dt_regressor.h"
#include "btdtr_ptz_util.h"
#include "ptz_pose_estimation.h"


using namespace::std;
extern "C" {
    // Create a model from a list of feature_label files
	EXPORTIT void createMap(const char * feature_label_file,
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
        tree_param.printSelf();
        
        RFMapBuilder builder;
        builder.setTreeParameter(tree_param);
        
        BTDTRegressor model;
        builder.buildModel(model, feature_files, model_name);
        model.saveModel(model_name);
    }
    // Extend the map by add a new tree
    // the new tree is built from feature_label files
	EXPORTIT void extendMap(const char* model_name,
                   const char* prev_feature_label_file,
                   const char* feature_label_file)
    {
        
    }
    
    // relocalize a camera using the model
    // parameter_file: testing parameter
    // pan_tilt_zoom: output
	EXPORTIT void relocalizeCamera(const char* model_name,
                          const char* feature_location_file_name,
                          const char* test_parameter_file,
                          double* pan_tilt_zoom)
    {
        //printf("%s %s %s\n", model_name, feature_location_file_name, test_parameter_file);
        // read model
        BTDTRegressor model;
        bool is_read = model.load(model_name);
        assert(is_read);
        
        // @todo read testing parameter
        
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
        Eigen::Vector3d estimated_ptz(0, 0, 0);        
        // predict from observation (descriptors)
        double tt = clock();
        for (int j = 0; j<samples.size(); j++) {
            btdtr_ptz_util::PTZSample s = samples[j];
            Eigen::VectorXf feat = s.descriptor_;
            vector<Eigen::VectorXf> cur_predictions;
            vector<float> cur_dists;
            model.predict(feat, max_check, cur_predictions, cur_dists);
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
                candidate_pan_tilt.push_back(cur_candidate);
            }
        }
        // estimate camera pose
        bool is_opt = ptz_pose_opt::preemptiveRANSACOneToMany(image_points, candidate_pan_tilt, pp.cast<double>(),
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
}
