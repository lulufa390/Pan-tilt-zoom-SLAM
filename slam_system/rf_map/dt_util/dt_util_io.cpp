//
//  DTUtil_IO.cpp
//  Classifer_RF
//
//  Created by jimmy on 2016-10-04.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "dt_util_io.hpp"

/********         DTUtil_IO           ************/
bool DTUtil_IO::read_matrix(const char * file_name, vector<VectorXd> & data)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("can not load from %s\n", file_name);
        return false;
    }
    assert(pf);
    int rows = 0;
    int cols = 0;
    int num = fscanf(pf, "%d %d", &rows, &cols);
    assert(num == 2);
    for (int i = 0; i<rows; i++) {
        VectorXd feat = VectorXd::Zero(cols);
        double val = 0;
        for (int j = 0; j<cols; j++) {
            num = fscanf(pf, "%lf", & val);
            assert(num == 1);
            feat[j] = val;
        }
        data.push_back(feat);
    }
    fclose(pf);
    printf("read data: %lu %lu \n", data.size(), data[0].size());
    return true;
}

bool DTUtil_IO::read_matrix(const char * file_name, vector<Eigen::VectorXf>& data)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("can not load from %s\n", file_name);
        return false;
    }
    assert(pf);
    int rows = 0;
    int cols = 0;
    int num = fscanf(pf, "%d %d", &rows, &cols);
    assert(num == 2);
    for (int i = 0; i<rows; i++) {
        VectorXf feat = VectorXf::Zero(cols);
        double val = 0;
        for (int j = 0; j<cols; j++) {
            num = fscanf(pf, "%lf", & val);
            assert(num == 1);
            feat[j] = val;
        }
        data.push_back(feat);
    }
    fclose(pf);
    printf("read data: %lu %lu \n", data.size(), data[0].size());
    return true;
}

bool DTUtil_IO::read_matrix(const char * file_name, Eigen::MatrixXd & data)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("can not load from %s\n", file_name);
        return false;
    }
    assert(pf);
    int rows = 0;
    int cols = 0;
    int num = fscanf(pf, "%d %d", &rows, &cols);
    assert(num == 2);
    data = Eigen::MatrixXd::Zero(rows, cols);
    for (int i = 0; i<rows; i++) {
        double val = 0;
        for (int j = 0; j<cols; j++) {
            num = fscanf(pf, "%lf", & val);
            assert(num == 1);
            data(i, j) = val;
        }
    }
    fclose(pf);
    printf("read data: %lu %lu \n", data.rows(), data.cols());
    return true;
}

bool DTUtil_IO::read_fn_matrix(const char *file_name, vector<int> & fns, vector<Eigen::VectorXd> & features)
{
    assert(fns.size() == 0);
    assert(features.size() == 0);
    
    vector<Eigen::VectorXd> fn_data;
    bool is_read = DTUtil_IO::read_matrix(file_name, fn_data);
    assert(is_read);
    assert(fn_data[0].size() > 1);
    
    int feat_size = (int)fn_data[0].size() - 1;
    for (int i = 0; i<fn_data.size(); i++) {
        // treat first column as frame number
        Eigen::VectorXd cur_feat = Eigen::VectorXd::Zero(feat_size);
        fns.push_back((int)fn_data[i][0]);
        // the rest column as feature
        for (int j = 1; j<fn_data[i].size(); j++) {
            cur_feat[j-1] = fn_data[i][j];
        }
        features.push_back(cur_feat);
    }
    assert(features.size() == fns.size());
    return true;
}

bool DTUtil_IO::read_fn_matrix(const char *file_name, vector<int> & fns, vector<Eigen::VectorXf> & features)
{
    assert(fns.size() == 0);
    assert(features.size() == 0);
    
    vector<Eigen::VectorXd> fn_data;
    bool is_read = DTUtil_IO::read_matrix(file_name, fn_data);
    assert(is_read);
    assert(fn_data[0].size() > 1);
    
    int feat_size = (int)fn_data[0].size() - 1;
    for (int i = 0; i<fn_data.size(); i++) {
        // treat first column as frame number
        Eigen::VectorXf cur_feat = Eigen::VectorXf::Zero(feat_size);
        fns.push_back((int)fn_data[i][0]);
        // the rest column as feature
        for (int j = 1; j<fn_data[i].size(); j++) {
            cur_feat[j-1] = fn_data[i][j];
        }
        features.push_back(cur_feat);
    }
    assert(features.size() == fns.size());
    return true;
}

bool DTUtil_IO::read_fn_labels(const char * file_name, vector<int> & fns, vector<unsigned int> & labels)
{
    assert(fns.size() == 0);
    assert(labels.size() == 0);
    
    vector<Eigen::VectorXd> fn_data;
    bool is_read = DTUtil_IO::read_matrix(file_name, fn_data);
    assert(is_read);
    assert(fn_data[0].size() == 2);
    
    for (int i = 0; i<fn_data.size(); i++) {
        int fn = (int)fn_data[i][0];
        unsigned int label = (unsigned int)fn_data[i][1];
        fns.push_back(fn);
        labels.push_back(label);
    }
    assert(fns.size() == labels.size());
    return true;
}

bool DTUtil_IO::read_fn_gd_preds(const char *file_name, vector<int> & fns, vector<unsigned int> & gds,  vector<unsigned int> & preds)
{
    assert(fns.size() == 0);
    assert(gds.size() == 0);
    assert(preds.size() == 0);
    
    vector<Eigen::VectorXd> fn_data;
    bool is_read = DTUtil_IO::read_matrix(file_name, fn_data);
    assert(is_read);
    assert(fn_data[0].size() == 3);
    
    for (int i = 0; i<fn_data.size(); i++) {
        int fn = (int)fn_data[i][0];
        unsigned int gd = (unsigned int)fn_data[i][1];
        unsigned int pred = (unsigned int)fn_data[i][2];
        fns.push_back(fn);
        gds.push_back(gd);
        preds.push_back(pred);
    }
    assert(fns.size() == gds.size());
    assert(fns.size() == preds.size());
    return true;
}

void DTUtil_IO::load_vertical_concat_feature_label(const vector<string> & feature_files, const vector<string> & label_files,
                                                   vector<Eigen::MatrixXf> & features, vector<Eigen::VectorXf> & labels)
{
    // read feature/label file names
    assert(feature_files.size() > 1);
    assert(label_files.size() > 0);
    
    // temporary data holder
    vector<vector<Eigen::VectorXd> > feature_groups(feature_files.size());
    vector<int> prev_input_fns;
    for (size_t i = 0; i<feature_files.size(); i++) {
        vector<int> input_fns;
        DTUtil_IO::read_fn_matrix(feature_files[i].c_str(), input_fns, feature_groups[i]);
        if (prev_input_fns.size() != 0) {
            assert(prev_input_fns.size() == input_fns.size());
            for (int j = 0; j<input_fns.size(); j++) {
                assert(input_fns[j] == prev_input_fns[j]);
            }
        }
        prev_input_fns = input_fns;
    }
    assert(feature_groups.size() == feature_files.size());
    
    // read labels
    vector<vector<Eigen::VectorXd> > label_groups(label_files.size());
    vector<int> prev_output_fns;
    for (size_t i = 0; i<label_files.size(); i++) {
        vector<int> output_fns;
        DTUtil_IO::read_fn_matrix(label_files[i].c_str(), output_fns, label_groups[i]);
        assert(label_groups[i].front().size() == 1);
        prev_output_fns = output_fns;
    }
    assert(prev_input_fns.size() == prev_output_fns.size());
    
    // check frame number
    for (int j = 0; j<prev_input_fns.size(); j++) {
        assert(prev_input_fns[j] == prev_output_fns[j]);
    }
    
    // assumen the frame number between channles are correct
    const int rows = (int)feature_groups.size();   // row is the same as channel
    const int cols = (int)feature_groups[0][0].size();
    const int N = (int)feature_groups[0].size();
    features.resize(N);
    labels.resize(N);
    
    for (int i = 0; i<N; i++) {
        MatrixXf feat = MatrixXf::Zero(rows, cols);
        for (int r = 0; r<rows; r++) {
            for (int c = 0; c<cols; c++) {
                feat(r, c) = feature_groups[r][i][c];
            }
        }
        
        features[i] = feat;
    }
    
    const int label_rows = (int)label_groups.size();
    for (int i = 0; i<N; i++) {
        VectorXf label = VectorXf::Zero(label_rows, 1);
        for (int r = 0; r<label_rows; r++) {
            label[r] = label_groups[r][i][0];
        }
        labels[i] = label;
    }
    
    assert(features.size() == labels.size());
}

void DTUtil_IO::load_vertical_concat_feature_label(const vector<string> & feature_files, const char * label_file,
                                                   vector<Eigen::MatrixXf> & features, vector<unsigned int> & labels)
{
    // read feature/label file names    
    assert(feature_files.size() > 1);
    
    // temporary data holder
    vector<vector<Eigen::VectorXd> > feature_groups(feature_files.size());
    
    // read feature files
    vector<int> prev_input_fns;
    for (size_t i = 0; i<feature_files.size(); i++) {
        vector<int> input_fns;
        bool is_read = DTUtil_IO::read_fn_matrix(feature_files[i].c_str(), input_fns, feature_groups[i]);
        assert(is_read);
        // check frame number
        if (prev_input_fns.size() != 0) {
            assert(prev_input_fns.size() == input_fns.size());
            for (int j = 0; j<input_fns.size(); j++) {
                assert(input_fns[j] == prev_input_fns[j]);
            }
        }
        prev_input_fns = input_fns;
    }
    
    // assumen the frame number between channles are correct
    const int rows = (int)feature_groups.size();   // row is the same as channel
    const int cols = (int)feature_groups[0][0].size();
    const int N = (int)feature_groups[0].size();
    features.resize(N);
    
    for (int i = 0; i<N; i++) {
        MatrixXf feat = MatrixXf::Zero(rows, cols);
        for (int r = 0; r<rows; r++) {
            for (int c = 0; c<cols; c++) {
                feat(r, c) = feature_groups[r][i][c];
            }
        }
        
        features[i] = feat;
    }    
    
    // read labels
    vector<int> label_fns;
    bool is_read = DTUtil_IO::read_fn_labels(label_file, label_fns, labels);
    assert(is_read);
    assert(label_fns.size() == prev_input_fns.size());
    for (int i = 0; i<label_fns.size(); i++) {
        assert(label_fns[i] ==  prev_input_fns[i]);
    }
    assert(features.size() == labels.size());
}


bool DTUtil_IO::save_matrix(const char * file_name, const vector<VectorXd> & data)
{
    assert(data.size() > 0);
    FILE *pf = fopen(file_name, "w");
    if (!pf) {
        printf("can not write to %s\n", file_name);
        return false;
    }
    assert(pf);
    fprintf(pf, "%d %d\n", (int)data.size(), (int)data[0].size());
    for (int i = 0; i<data.size(); i++) {
        for (int j = 0; j<data[i].size(); j++) {
            fprintf(pf, "%lf ", data[i][j]);
            if (j == data[i].size()-1) {
                fprintf(pf, "\n");
            }
        }
    }
    printf("save to %s\n", file_name);
    return true;
}

bool DTUtil_IO::read_labels(const char * file_name, vector<unsigned int> & labels)
{
    vector<Eigen::VectorXd> data;
    bool is_read = DTUtil_IO::read_matrix(file_name, data);
    assert(is_read);
    assert(data[0].size() == 1);
    for (int i = 0; i<data.size(); i++) {
        int val = (unsigned int)data[i][0];
        labels.push_back(val);
    }
    return true;
}

bool DTUtil_IO::read_files(const char *file_name, vector<string> & files)
{
    FILE *pf = fopen(file_name, "r");
    if (!pf) {
        printf("can not load from %s\n", file_name);
        return false;
    }
    assert(pf);
    while (1) {
        char line[1024] = {NULL};
        int ret = fscanf(pf, "%s", line);
        if (ret != 1) {
            break;
        }
        files.push_back(string(line));
    }
    printf("read %lu lines\n", files.size());
    fclose(pf);
    return true;
}

bool DTUtil_IO::write_files(const char *file_name, const vector<string>& files)
{
    FILE *pf = fopen(file_name, "w");
    if (!pf) {
        printf("can not write a file %s\n", file_name);
        return false;
    }
    for (int i = 0; i<files.size(); i++) {
        fprintf(pf, "%s\n", files[i].c_str());
    }
    printf("write %lu lines\n", files.size());
    fclose(pf);
    
    return true;
}

template <class T_VectorX>
bool DTUtil_IO::saveVectorsAsMatrix(const char * file_name, const vector<T_VectorX> & data)
{
    assert(data.size() > 0);
    FILE *pf = fopen(file_name, "w");
    if (!pf) {
        printf("can not write to %s\n", file_name);
        return false;
    }
    assert(pf);
    fprintf(pf, "%d %d\n", (int)data.size(), (int)data[0].size());
    for (int i = 0; i<data.size(); i++) {
        for (int j = 0; j<data[i].size(); j++) {
            fprintf(pf, "%lf ", (double)data[i][j]);
            if (j == data[i].size()-1) {
                fprintf(pf, "\n");
            }
        }
    }
    printf("save to %s\n", file_name);
    return true;
}

// instantiate
template bool
DTUtil_IO::saveVectorsAsMatrix(const char * file_name, const vector<Eigen::VectorXf> & data);


