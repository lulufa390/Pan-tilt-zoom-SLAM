//
//  yael_io.cpp
//  OLNN
//
//  Created by jimmy on 2016-12-02.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include <stdio.h>
#include "yael_io.h"


static long xvecs_fsize(long unitsize, const char * fname, int *d_out, int *n_out)
{
    int d, ret;
    long nbytes;
    
    *d_out = -1;
    *n_out = -1;
    
    FILE * f = fopen (fname, "r");
    
    if(!f) {
        fprintf(stderr, "xvecs_fsize %s: %s\n", fname, strerror(errno));
        return -1;
    }
    /* read the dimension from the first vector */
    ret = (int)fread (&d, sizeof (d), 1, f);
    if (ret == 0) { /* empty file */
        *n_out = 0;
        return ret;
    }
    
    fseek (f, 0, SEEK_END);
    nbytes = ftell (f);
    fclose (f);
    
    if(nbytes % (unitsize * d + 4) != 0) {
        fprintf(stderr, "xvecs_size %s: weird file size %ld for vectors of dimension %d\n", fname, nbytes, d);
        return -1;
    }
    
    *d_out = d;
    *n_out = (int)(nbytes / (unitsize * d + 4));
    return nbytes;
}


float *YaelIO::fvec_new (long n)
{
    float *buf = NULL;
    int error_msg = posix_memalign ((void **)&buf, 16, sizeof (*buf) * n);
    if (error_msg == EINVAL || error_msg == ENOMEM) {
        fprintf (stderr, "fvec_new %ld : out of memory or not aligned \n", n);
        abort();
    }
    return buf;
}

int *YaelIO::ivec_new(long n)
{
    int *buf = NULL;
    int error_msg = posix_memalign ((void **)&buf, 16, sizeof (*buf) * n);
    if (error_msg == EINVAL || error_msg == ENOMEM) {
        fprintf (stderr, "fvec_new %ld : out of memory or not aligned \n", n);
        abort();
    }
    return buf;   
}

bool YaelIO::read_fvecs_file(const char *file_name, Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(file_name);
    
    int d = 0;
    int n = 0;
    int ret = (int)YaelIO::fvecs_fsize (file_name, &d, &n);
    assert(ret != -1);
    fprintf (stderr, "File %s contains %d vectors of dimension %d\n", file_name, n, d);
    
    float *v = YaelIO::fvec_new(n * d);
    assert(v != NULL);
    ret = YaelIO::fvecs_read(file_name, d, n, v);
    assert(ret == n);
    
    data = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(n, d);
    
    //! can be speed up by memory mapping
    for (int i = 0; i<n; i++) {
        for (int j = 0; j<d; j++) {
            data(i, j) = v[i*d + j];
        }
    }
    free(v);
    return true;
}

bool YaelIO::read_ivecs_file(const char *file_name, Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(file_name);
    
    int d = 0;
    int n = 0;
    int ret = (int)YaelIO::ivecs_fsize (file_name, &d, &n);
    fprintf (stderr, "File %s contains %d vectors of dimension %d\n", file_name, n, d);
    
    int *v = YaelIO::ivec_new(n * d);
    assert(v != NULL);
    ret = YaelIO::ivecs_read(file_name, d, n, v);
    assert(ret == n);
    
    data = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(n, d);
    
    //! can be speed up by memory mapping
    for (int i = 0; i<n; i++) {
        for (int j = 0; j<d; j++) {
            data(i, j) = v[i*d + j];
        }
    }
    free(v);
    
    return true;
}

bool YaelIO::write_fvecs_file(const char *file_name, const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(file_name);
    FILE *pf = fopen(file_name, "w");
    if (!pf) {
        perror ("ivecs_write");
        return false;
    }
    
    int d = (int)data.cols();
    int n = (int)data.rows();
    const float * v = data.data();
    fvecs_fwrite (pf, d, n, v);
    
    fclose(pf);    
    return true;
}

bool YaelIO::write_ivecs_file(const char *file_name, const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & data)
{
    assert(file_name);
    FILE *pf = fopen(file_name, "w");
    if (!pf) {
        perror ("ivecs_write");
        return false;
    }
    
    int d = (int)data.cols();
    int n = (int)data.rows();
    const int * v = data.data();
    ivecs_fwrite (pf, d, n, v);
    
    fclose(pf);
    return true;
}





long YaelIO::fvecs_fsize (const char * fname, int *d_out, int *n_out) {
    return xvecs_fsize(sizeof(float), fname, d_out, n_out);
}

long YaelIO::ivecs_fsize (const char * fname, int *d_out, int *n_out)
{
    return xvecs_fsize (sizeof(int), fname, d_out, n_out);
}

long YaelIO::bvecs_fsize (const char * fname, int *d_out, int *n_out)
{
    return xvecs_fsize (sizeof(unsigned char), fname, d_out, n_out);
}

long YaelIO::lvecs_fsize (const char * fname, int *d_out, int *n_out)
{
    return xvecs_fsize (sizeof(long long), fname, d_out, n_out);
}






int YaelIO::fvecs_read (const char *fname, int d, int n, float *a)
{
    FILE *f = fopen (fname, "r");
    if (!f) {
        fprintf (stderr, "fvecs_read: could not open %s\n", fname);
        perror ("");
        return -1;
    }
    
    long i;
    for (i = 0; i < n; i++) {
        int new_d;
        
        if (fread (&new_d, sizeof (int), 1, f) != 1) {
            if (feof (f))
            break;
            else {
                perror ("fvecs_read error 1");
                fclose(f);
                return -1;
            }
        }
        
        if (new_d != d) {
            fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
            fclose(f);
            return -1;
        }
        
        if (fread (a + d * (long) i, sizeof (float), d, f) != d) {
            fprintf (stderr, "fvecs_read error 3\n");
            fclose(f);
            return -1;
        }
    }
    fclose (f);
    return (int)i;
}

int YaelIO::ivecs_read(const char *fname, int d, int n, int *a)
{
    FILE *f = fopen (fname, "r");
    if (!f) {
        fprintf (stderr, "fvecs_read: could not open %s\n", fname);
        perror ("");
        return -1;
    }
    
    long i;
    for (i = 0; i < n; i++) {
        int new_d;
        
        // Each vector takes 4+d*4 bytes for .fvecs and .ivecs formats, and 4+d bytes for .bvecs formats
        if (fread (&new_d, sizeof (int), 1, f) != 1) {
            if (feof (f))
            break;
            else {
                perror ("ivecs_read error 1");
                fclose(f);
                return -1;
            }
        }
        
        if (new_d != d) {
            fprintf (stderr, "ivecs_read error 2: unexpected vector dimension\n");
            fclose(f);
            return -1;
        }
        
        if (fread (a + d * (long) i, sizeof (int), d, f) != d) {
            fprintf (stderr, "fvecs_read error 3\n");
            fclose(f);
            return -1;
        }
    }
    fclose (f);
    return (int)i;
}


int YaelIO::ivec_fwrite (FILE *f, const int *v, int d)
{
    int ret = (int)fwrite (&d, sizeof (d), 1, f);
    if (ret != 1) {
        perror ("ivec_fwrite: write error 1");
        return -1;
    }
    
    ret = (int)fwrite (v, sizeof (*v), d, f);
    if (ret != d) {
        perror ("ivec_fwrite: write error 2");
        return -2;
    }
    return 0;
}

int YaelIO::ivecs_fwrite(FILE *f, int d, int n, const int *v)
{
    int i;
    for (i = 0 ; i < n ; i++) {
        ivec_fwrite (f, v, d);
        v+=d;
    }
    return n;
}

int YaelIO::fvec_fwrite (FILE *fo, const float *v, int d)
{
    int ret;
    ret = (int)fwrite (&d, sizeof (int), 1, fo);
    if (ret != 1) {
        perror ("fvec_fwrite: write error 1");
        return -1;
    }
    ret = (int)fwrite (v, sizeof (float), d, fo);
    if (ret != d) {
        perror ("fvec_fwrite: write error 2");
        return -1;
    }  
    return 0;
}


int YaelIO::fvecs_fwrite (FILE *fo, int d, int n, const float *vf)
{
    int i;
    /* write down the vectors as fvecs */
    for (i = 0; i < n; i++) {
        if(fvec_fwrite(fo, vf+i*d, d)<0)
            return i;
    }
    return n;
}



void YaelIOUtil::load_1k_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                 Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth)
{
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1k/sift1k_base.fvecs", base_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1k/sift1k_learn.fvecs", learn_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1k/sift1k_query.fvecs", query_data);
    YaelIO::read_ivecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1k/sift1k_groundtruth.ivecs", ground_truth);
}

void YaelIOUtil::load_1k_reorder_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                         Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                         Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                         Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth)
{
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1k_reorder/sift1k_base.fvecs", base_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1k_reorder/sift1k_learn.fvecs", learn_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1k_reorder/sift1k_query.fvecs", query_data);
    YaelIO::read_ivecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1k_reorder/sift1k_groundtruth.ivecs", ground_truth);
    
}

void YaelIOUtil::load_10k_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                      Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth)
{
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/siftsmall/siftsmall_base.fvecs", base_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/siftsmall/siftsmall_learn.fvecs", learn_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/siftsmall/siftsmall_query.fvecs", query_data);
    YaelIO::read_ivecs_file("/Users/jimmy/Desktop/learning_data/ANN/siftsmall/siftsmall_groundtruth.ivecs", ground_truth);
}



void YaelIOUtil::load_10k_32dim_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth)
{
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift10k_32_dim/sift10k_32d_base.fvecs", base_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift10k_32_dim/sift10k_32d_learn.fvecs", learn_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift10k_32_dim/sift10k_32d_query.fvecs", query_data);
    YaelIO::read_ivecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift10k_32_dim/sift10k_32d_groundtruth.ivecs", ground_truth);
}

void YaelIOUtil::load_100k_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                   Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth)
{
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift100k/sift100k_base.fvecs", base_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift100k/sift100k_learn.fvecs", learn_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift100k/sift100k_query.fvecs", query_data);
    YaelIO::read_ivecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift100k/sift100k_groundtruth.ivecs", ground_truth);
    
}

void YaelIOUtil::load_1m_dataset(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & base_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & learn_data,
                                 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & query_data,
                                 Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> & ground_truth)
{
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1m/sift_base.fvecs", base_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1m/sift_learn.fvecs", learn_data);
    YaelIO::read_fvecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1m/sift_query.fvecs", query_data);
    YaelIO::read_ivecs_file("/Users/jimmy/Desktop/learning_data/ANN/sift1m/sift_groundtruth.ivecs", ground_truth);
}









