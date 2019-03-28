//
//  cvxUtil.cpp
//  RGB_RF
//
//  Created by jimmy on 2016-05-27.
//  Copyright © 2016 jimmy. All rights reserved.
//

#include "cvx_util.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

#include <dirent.h>
#include <string.h>


vector<double>
CvxUtil:: generateRandomNumbers(double min_val, double max_val, int rnd_num)
{
    assert(rnd_num > 0);
    
    cv::RNG rng;
    vector<double> data;
    for (int i = 0; i<rnd_num; i++) {
        data.push_back(rng.uniform(min_val, max_val));
    }
    return data;
}

void
CvxUtil::splitFilename (const string& str, string &path, string &file)
{
    assert(!str.empty());
    unsigned int found = (unsigned int )str.find_last_of("/\\");
    path = str.substr(0, found);
    file = str.substr(found + 1);
}

void CvxUtil::readFilenames(const char *folder, vector<string> & file_names)
{
    const char *post_fix = strrchr(folder, '.');
    string pre_str(folder);
    pre_str = pre_str.substr(0, pre_str.rfind('/') + 1);
    //printf("pre_str is %s\n", pre_str.c_str());
    
    assert(post_fix);
    // vcl_vector<vcl_string> file_names;
    DIR *dir = NULL;
    struct dirent *ent = NULL;
    if ((dir = opendir (pre_str.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            const char *cur_post_fix = strrchr( ent->d_name, '.');
            if (!cur_post_fix ) {
                continue;
            }
            //printf("cur post_fix is %s %s\n", post_fix, cur_post_fix);
            
            if (!strcmp(post_fix, cur_post_fix)) {
                file_names.push_back(pre_str + string(ent->d_name));
                //  cout<<file_names.back()<<endl;
            }
            
            //printf ("%s\n", ent->d_name);
        }
        closedir (dir);
    }
    printf("read %lu files\n", file_names.size());
}



unsigned
CvxUtil::value_to_bin_number(double v_min, double interval, double value, const unsigned nBin)
{
    int num = (value - v_min)/interval;
    if (num < 0) {
        return 0;
    }
    if (num >= nBin) {
        return nBin - 1;
    }
    return (unsigned)num;
}

double
CvxUtil::bin_number_to_value(double v_min, double interval, int bin)
{
    return v_min + bin * interval;
}

