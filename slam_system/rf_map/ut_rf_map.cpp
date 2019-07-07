//
//  ut_rf_map.cpp
//  ptz_slam_dev
//
//  Created by jimmy on 2019-07-04.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "ut_rf_map.hpp"
#include "rf_map.hpp"
#include "online_rf_map_builder.hpp"
#include <fstream>

using namespace std;

void ut_rf_map()
{
    string tree_param_file("/Users/jimmy/Desktop/basketball_standard_rf/ptz_tree_param.txt");
    string featue_label_files = "/Users/jimmy/Desktop/basketball_standard_rf/train_file.txt";
    string feature_location_file_name("/Users/jimmy/Desktop/basketball_standard_rf/keyframes/664.mat");
    
    RFMap* rf_map = RFMap_new();
    
    createMap(rf_map, featue_label_files.c_str(), tree_param_file.c_str(), "tree_debug.txt");
    
    double ptz[3] = {6.32164, -10.0, 2436.07};
    relocalizeCamera(rf_map, feature_location_file_name.c_str(), "", ptz);
    printf("refined pan, tilt zoom %f %f %f\n", ptz[0], ptz[1], ptz[2]);
    
    RFMap_delete(rf_map);
}




void ut_update_tree()
{
    
    string tree_param_file("/Users/jimmy/Desktop/basketball_standard_rf/ptz_tree_param_2.txt");
    string online_train_file_1 = "/Users/jimmy/Desktop/basketball_standard_rf/online_train_file_1.txt";
    string online_train_file_2 = "/Users/jimmy/Desktop/basketball_standard_rf/online_train_file_2.txt";
    
    vector<string> feature_label_files;
    ifstream file(online_train_file_1);
    string str;
    while(std::getline(file, str)) {
        feature_label_files.push_back(str);
    }
    printf("read %lu feature label files\n", feature_label_files.size());
    
    OnlineRFMapBuilder builder;
    
    btdtr_ptz_util::PTZTreeParameter tree_param;
    tree_param.readFromFile(tree_param_file.c_str());
    builder.setTreeParameter(tree_param);
    
    // initial tree
    BTDTRegressor model;
    builder.setTreeParameter(tree_param);
    builder.addTree(model, feature_label_files, "online_tree_debug.txt");
    
    
    //string feature_location_file_name("/Users/jimmy/Desktop/basketball_standard_rf/keyframes/664.mat");
    //double ptz[3] = {6.32164, -10.0, 2436.07};
    //relocalizeCamera(rf_map, feature_location_file_name.c_str(), "", ptz);
    //printf("refined pan, tilt zoom %f %f %f\n", ptz[0], ptz[1], ptz[2]);
    
    // updated tree
    string feature_label_file("/Users/jimmy/Desktop/basketball_standard_rf/keyframes/737.mat");
    vector<float> errors;
    bool is_add_tree = builder.isAddTree(model, feature_label_file, 0.1, 0.5);
    if (is_add_tree) {
        printf("add a tree\n");
    }
    else {
        printf("update a tree\n");
    }
    builder.updateTree(model, feature_label_file, "online_tree_debug_2.txt");
}

void ut_add_or_add_tree()
{
    string tree_param_file("/Users/jimmy/Desktop/basketball_standard_rf/ptz_tree_param_2.txt");
    string train_files = "/Users/jimmy/Desktop/basketball_standard_rf/train_file.txt";
    
    vector<string> feature_label_files;
    ifstream file(train_files);
    string str;
    while(std::getline(file, str)) {
        feature_label_files.push_back(str);
    }
    printf("read %lu feature label files\n", feature_label_files.size());
    
    OnlineRFMapBuilder builder;
    
    btdtr_ptz_util::PTZTreeParameter tree_param;
    tree_param.readFromFile(tree_param_file.c_str());
    builder.setTreeParameter(tree_param);
    
    // add first tree
    vector<string> first_file(feature_label_files.begin(), feature_label_files.begin()+1);
    
    BTDTRegressor model;
    builder.setTreeParameter(tree_param);
    builder.addTree(model, first_file, "online_tree_debug.txt");
    
    // dynamically add or update tree
    for (int i = 1; i<feature_label_files.size(); i++) {
        string cur_file = feature_label_files[i];
        bool is_add = builder.isAddTree(model, cur_file, 0.1, 0.5);
        if (is_add) {
            printf("------------- add a tree ---------------\n");
            builder.addTree(model, cur_file, "online_tree_debug.txt", false);
        }
        else {
            printf("------------- update a tree ---------------\n");
            builder.updateTree(model, cur_file, "online_tree_debug.txt", false);
        }
    }
    
}
