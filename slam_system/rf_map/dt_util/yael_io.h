//
//  yael_io.h
//  OLNN
//
//  Created by jimmy on 2016-12-01.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef OLNN_yael_io_h
#define OLNN_yael_io_h

// file IO for http://corpus-texmex.irisa.fr/

#include <stdio.h>
#include <Eigen/Dense>
#include <assert.h>

using Eigen::MatrixXf;
using Eigen::Matrix;

class YaelIO
{
public:
    // n: number
    // d: dimension
    static bool read_fvecs_file(const char *file_name, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    static bool read_ivecs_file(const char *file_name, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    
    // .fvecs
    static bool write_fvecs_file(const char *file_name, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    static bool write_ivecs_file(const char *file_name, const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data);
    
private:
    static float *fvec_new (long n);
    static int *ivec_new(long n);
    
    static int fvecs_read(const char *fname, int d, int n, float *v);
    static int ivecs_read(const char *fname, int d, int n, int *v);
    
    /*!  Read the number of vectors in a file and their dimension
     (vectors of same size). Output the number of bytes of the file. */
    static long fvecs_fsize (const char * fname, int *d_out, int *n_out);
    static long ivecs_fsize (const char * fname, int *d_out, int *n_out);
    static long bvecs_fsize (const char * fname, int *d_out, int *n_out);
    static long lvecs_fsize (const char * fname, int *d_out, int *n_out);
    
    
    /*!  write a set of vectors into an open file */
    static int ivecs_fwrite(FILE *f, int d, int n, const int *v);
    static int fvecs_fwrite(FILE *fo, int d, int n, const float *vf);
    
    /*!  write a vector into an open file */
    static int ivec_fwrite(FILE *f, const int *v, int d);
    static int fvec_fwrite(FILE *f, const float *v, int d);
};



class YaelIOUtil
{
public:
    static void load_1k_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth);
    
    static void load_1k_reorder_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth);

    
    static void load_10k_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                 Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth);
    
    static void load_10k_32dim_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                 Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth);
    
    static void load_100k_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                 Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth);
    
    static void load_1m_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                 Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth);
    
};


#endif
