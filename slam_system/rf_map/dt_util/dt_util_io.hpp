//
//  DTUtil_IO.h
//  Classifer_RF
//
//  Created by jimmy on 2016-10-04.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__DTUtil_IO__
#define __Classifer_RF__DTUtil_IO__

#include <stdio.h>
#include <vector>
#include <Eigen/Dense>
#include <string>
#include <fstream>

using std::vector;
using std::string;
using Eigen::VectorXd;
using Eigen::VectorXf;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using Eigen::MatrixXi;

class DTUtil_IO
{
public:
    static bool read_matrix(const char * file_name, vector<VectorXd> & data);
    static bool read_matrix(const char * file_name, vector<Eigen::VectorXf>& data);
    static bool read_matrix(const char * file_name, Eigen::MatrixXd & data);
    static bool save_matrix(const char * file_name, const vector<VectorXd> & data);    
    static bool read_labels(const char * file_name, vector<unsigned int> & labels);
    
    // files with frame number as the first column
    static bool read_fn_matrix(const char *file_name, vector<int> & fns, vector<Eigen::VectorXd> & data);
    static bool read_fn_matrix(const char *file_name, vector<int> & fns, vector<Eigen::VectorXf> & data);
    static bool read_fn_labels(const char * file_name, vector<int> & fns, vector<unsigned int> & labels);
    static bool read_fn_gd_preds(const char *file_name, vector<int> & fns, vector<unsigned int> & gds,  vector<unsigned int> & preds);
    
    // file with frame number as the first column
    // original features and labels are synchronized (e.g., frame numbers)
    // original label dimension is 1
    static void load_vertical_concat_feature_label(const vector<string> & feature_files, const vector<string> & label_files,
                                                   vector<Eigen::MatrixXf> & features, vector<Eigen::VectorXf> & labels);
    
    // multiple feature files and a single label file
    static void load_vertical_concat_feature_label(const vector<string> & feature_files, const char * label_file,
                                                   vector<Eigen::MatrixXf> & features, vector<unsigned int> & labels);
    
    
    
    static bool read_files(const char *file_name, vector<string> & files);
    static bool write_files(const char *file_name, const vector<string>& files);
    
    //
    template<class T>
    static bool save_matrix(const char * file_name, const T &m)
    {
        std::ofstream file(file_name);
        if (file.is_open()) {
            int rows = (int)m.rows();
            int cols = (int)m.cols();
            file<<rows<<" "<<cols<<"\n"<<m<<"\n";
            printf("save to %s\n", file_name);
            return true;
        }
        else {
            printf("Error: can not open file %s\n", file_name);
            return false;
        }
    }
    
    template <class T_VectorX>
    static bool saveVectorsAsMatrix(const char * file_name, const vector<T_VectorX> & data);
    
    
};




#endif /* defined(__Classifer_RF__DTUtil_IO__) */
