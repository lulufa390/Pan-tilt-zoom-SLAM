//
//  mat_io.cpp
//  SequentialRandomForest
//
//  Created by jimmy on 2017-07-01.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "mat_io.hpp"
#ifdef __cplusplus
extern "C" {
    #include "matio.h"
#endif

#ifdef __cplusplus
}
#endif  // closing brace for extern "C"

using Eigen::Matrix;
using std::string;

namespace matio {
    
    template<class matrixT>
    bool readMatrix(const char *file_name, const char *var_name, matrixT & mat_data, bool verbose)
    {
        assert(file_name);
        assert(var_name);
        
        mat_t    *matfp = NULL;
        matvar_t *matvar = NULL;
        bool is_read = false;
        
        matfp = Mat_Open(file_name, MAT_ACC_RDONLY);
        if ( NULL == matfp ) {
            printf("Error: opening MAT file \"%s\"!\n", file_name);
            return false;
        }
        matvar = Mat_VarRead(matfp, var_name);
        if ( NULL == matvar ) {
            printf("Error: Variable %s not found, or error reading MAT file",
                    var_name);
        }
        if (matvar->rank != 2) {
            printf("Error: Variable %s is not a matrix!\n", var_name);
        }
        else {
            size_t rows = matvar->dims[0];
            size_t cols = matvar->dims[1];
            void *data = matvar->data;
            assert(data);
            matio_types data_type = matvar->data_type;
            
            switch (data_type) {
                case MAT_T_DOUBLE:
                {
                    mat_data = matrixT::Zero(rows, cols);
                    // colum wise
                    double *pdata = (double *)data;
                    // copy data
                    for (int c = 0; c<cols; c++ ) {
                        double * p = &pdata[c * rows];
                        for (int r = 0; r<rows; r++) {
                            mat_data(r, c) = p[r];
                        }
                    }
                    is_read = true;
                }                    
                break;
                    
                case MAT_T_SINGLE:
                {
                    mat_data = matrixT::Zero(rows, cols);
                    float *pdata = (float *)data;
                    // copy data
                    for (int c = 0; c<cols; c++ ) {
                        float * p = &pdata[c * rows];
                        for (int r = 0; r<rows; r++) {
                            mat_data(r, c) = p[r];
                        }
                    }

                    is_read = true;
                }
                break;
                    
                default:
                    printf("Error: non-supported data type\n");
                    break;
            }
        }
        
        // free data
        if (matvar != NULL) {
            Mat_VarFree(matvar);
            matvar = NULL;
        }
        if (matfp != NULL) {
            Mat_Close(matfp);
        }
        if (is_read && verbose) {
            printf("read a %ld x %ld matrix named %s. \n", mat_data.rows(), mat_data.cols(), var_name);
        }
        return is_read;
    }
    
    bool readString(const char *file_name, const char *var_name, std::string& output_data, bool verbose)
    {
        assert(file_name);
        assert(var_name);
        
        mat_t    *matfp = NULL;
        matvar_t *matvar = NULL;
        bool is_read = false;
        
        matfp = Mat_Open(file_name, MAT_ACC_RDONLY);
        if ( NULL == matfp ) {
            printf("Error: opening MAT file \"%s\"!\n", file_name);
            return false;
        }
        matvar = Mat_VarRead(matfp, var_name);
        if ( NULL == matvar ) {
            printf("Error: Variable %s not found, or error reading MAT file",
                   var_name);
        }
        if (matvar->class_type != MAT_C_CHAR) {
            printf("Error: Variable %s is not a char array!\n", var_name);
        }
        else {
            // string is 1 x N array
            size_t rows = matvar->dims[0];
            size_t cols = matvar->dims[1];
            void *data = matvar->data;
            assert(data);
            assert(rows == 1);
            assert(matvar->data_type == MAT_T_UTF8 ||
                   matvar->data_type == MAT_T_UINT8); // why MAT_T_UINT8?
            
            char *pdata = (char*)data;
            output_data.resize(cols);
            for (int i = 0; i<cols; i++) {
                output_data[i] = pdata[i];
            }
            is_read = true;
        }
        
        // free data
        if (matvar != NULL) {
            Mat_VarFree(matvar);
            matvar = NULL;
        }
        if (matfp != NULL) {
            Mat_Close(matfp);
        }
        
        if (is_read && verbose) {
            printf("read a %s from named %s. \n", output_data.c_str(), var_name);
        }
        
        return is_read;
    }
    
    template<class matrixT>
    bool writeMatrix(const char *file_name, const char *var_name, const matrixT& data)
    {
        assert(file_name);
        assert(var_name);
        mat_t    *matfp = NULL;
        matvar_t *matvar = NULL;
        bool is_write = false;
        const long rows = data.rows();
        const long cols = data.cols();
        size_t    dims[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
        
        matfp = Mat_CreateVer(file_name, NULL, MAT_FT_DEFAULT);
        if ( NULL == matfp ) {
            printf("Error: creating MAT file %s \n", file_name);
            return false;
        }
        
        // temporal data holder
        double* pdata = new double[rows * cols];
        assert(pdata);
        // colum wise copy data
        for (int r = 0; r<rows; r++) {
            for (int c = 0; c<cols; c++) {
                pdata[c*rows + r] = data(r, c);
            }
        }
        
        // only support double
        matvar = Mat_VarCreate(var_name, MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, (void*)pdata, 0);
        if ( NULL == matvar ) {
            fprintf(stderr,"Error creating variable for %s.\n", var_name);
            is_write = false;
        } else {
            Mat_VarWrite(matfp,matvar,MAT_COMPRESSION_NONE);
            Mat_VarFree(matvar);
            is_write = true;
        }
        Mat_Close(matfp);
        if (is_write) {
            printf("write a %ld x %ld matrix named %s to %s. \n", data.rows(), data.cols(),
                   var_name, file_name);
        }
        
        delete [] pdata;
        return is_write;
    }
    
    template<class matrixT>
    bool writeMultipleMatrix(const char *file_name,
                             const std::vector<std::string>& var_name,
                             const std::vector<matrixT>& data)
    {
        assert(file_name);
        assert(var_name.size() == data.size());
        assert(var_name.size() > 0);
                
        mat_t    *matfp = NULL;
        matvar_t *matvar = NULL;
        bool is_write = false;
        
        matfp = Mat_CreateVer(file_name, NULL, MAT_FT_DEFAULT);
        if ( NULL == matfp ) {
            printf("Error: creating MAT file %s \n", file_name);
            return false;
        }
        
        for (int i = 0; i<var_name.size(); i++) {
            const long rows = data[i].rows();
            const long cols = data[i].cols();
            size_t    dims[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
            double* pdata = new double[rows * cols];
            assert(pdata);
            
            // colum wise copy data
            for (int r = 0; r<rows; r++) {
                for (int c = 0; c<cols; c++) {
                    pdata[c*rows + r] = data[i](r, c);
                }
            }
            
            matvar = Mat_VarCreate(var_name[i].c_str(), MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, (void*)pdata, 0);
            if ( NULL == matvar ) {
                fprintf(stderr,"Error creating variable for %s.\n", var_name[i].c_str());
                is_write = false;
                if (pdata) {
                    delete []pdata;
                    pdata = NULL;
                }
                break;
            } else {
                Mat_VarWrite(matfp,matvar,MAT_COMPRESSION_NONE);
                Mat_VarFree(matvar);
                matvar = NULL;
                is_write = true;
                printf("write variable: %s\n", var_name[i].c_str());
            }
            if (pdata) {
                delete []pdata;
                pdata = NULL;
            }
        }
        Mat_Close(matfp);
        if (is_write) {
            printf("write %s. \n", file_name);
        }
        return is_write;
    }
    
    template<class matrixT>
    bool writeStringMatrix(const char *file_name,
                           const unordered_map<string, string>& strs,
                           const unordered_map<string, matrixT>& mats)
    {
        assert(file_name);
        assert(strs.size() > 0 || mats.size() > 0);
        
        mat_t    *matfp = NULL;
        matvar_t *matvar = NULL;
        bool is_write = false;
        
        // 1. open a file
        matfp = Mat_CreateVer(file_name, NULL, MAT_FT_DEFAULT);
        if ( NULL == matfp ) {
            printf("Error: creating MAT file %s \n", file_name);
            return false;
        }

        // 2. write strings
        for (const auto& item: strs) {
            const string& name = item.first;
            const string& data = item.second;
            size_t dims[2] = {1, static_cast<size_t>(data.size())};
            matvar = Mat_VarCreate(name.c_str(), MAT_C_CHAR, MAT_T_UTF8, 2, dims, (void*)data.c_str(), 0);
            if (NULL == matvar) {
                fprintf(stderr, "Error: creating variable for %s.\n", name.c_str());
                is_write = false;
                break;
            }
            else {
                Mat_VarWrite(matfp,matvar,MAT_COMPRESSION_NONE);
                Mat_VarFree(matvar);
                matvar = NULL;
                is_write = true;
                printf("write variable: %s\n", name.c_str());
            }
        }
        // 3. write matrix
        for (const auto& item: mats) {
            const string& name = item.first;
            const matrixT& data = item.second;
            
            const long rows = data.rows();
            const long cols = data.cols();
            size_t    dims[2] = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
            double* pdata = new double[rows * cols];
            assert(pdata);
            
            // colum wise copy data
            for (int r = 0; r<rows; r++) {
                for (int c = 0; c<cols; c++) {
                    pdata[c*rows + r] = data(r, c);
                }
            }
            
            matvar = Mat_VarCreate(name.c_str(), MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dims, (void*)pdata, 0);
            if ( NULL == matvar ) {
                fprintf(stderr, "Error: creating variable for %s.\n", name.c_str());
                is_write = false;
                if (pdata) {
                    delete []pdata;
                    pdata = NULL;
                }
                break;
            } else {
                Mat_VarWrite(matfp,matvar,MAT_COMPRESSION_NONE);
                Mat_VarFree(matvar);
                matvar = NULL;
                is_write = true;
                printf("write variable: %s\n", name.c_str());
            }
            if (pdata) {
                delete []pdata;
                pdata = NULL;
            }
        }
        
        // 4. close file and release data
        Mat_Close(matfp);
        if (is_write) {
            printf("write %s. \n", file_name);
        }
        return is_write;
    }
    
    template
    bool readMatrix(const char *file_name, const char *var_name, Eigen::MatrixXd& mat_data, bool verbose);
   
    template
    bool readMatrix(const char *file_name, const char *var_name, Eigen::MatrixXf& mat_data, bool verbose);
    
    template
    bool readMatrix(const char *file_name, const char *var_name, Eigen::MatrixXi& mat_data, bool verbose);
    
    template
    bool writeMatrix(const char *file_name, const char *var_name, const Eigen::MatrixXd& data);
    
    template
    bool writeMatrix(const char *file_name, const char *var_name, const Eigen::MatrixXf& data);
    
    template
    bool writeMultipleMatrix(const char *file_name,
                             const std::vector<std::string>& var_name,
                             const std::vector<Eigen::MatrixXd>& data);
    
    template
    bool writeMultipleMatrix(const char *file_name,
                             const std::vector<std::string>& var_name,
                             const std::vector<Eigen::MatrixXf>& data);
    
    template
    bool writeStringMatrix(const char *file_name,
                           const unordered_map<string, string>& strs,
                           const unordered_map<string, Eigen::MatrixXd>& mats);
    
    template
    bool writeStringMatrix(const char *file_name,
                           const unordered_map<string, string>& strs,
                           const unordered_map<string, Eigen::MatrixXf>& mats);
    
    
    
    
} // namespace matio
