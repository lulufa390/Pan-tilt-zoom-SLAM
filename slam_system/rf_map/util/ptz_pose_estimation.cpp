//
//  ptz_pose_estimation.cpp
//  PTZBTRF
//
//  Created by jimmy on 2017-08-09.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "ptz_pose_estimation.h"
#include "eigen_geometry_util.h"
#include "pgl_ptz_camera.h"
#include <iostream>

using std::cout;
using std::endl;

namespace ptz_pose_opt {
    namespace {
        struct Hypothesis
        {
            double loss_;
            Eigen::Vector3d ptz_;
            vector<int> inlier_indices_;         // image coordinate index
            vector<int> inlier_candidate_pan_tilt_indices_; // camera coordinate pan tilt index
            
            // store all inliers from preemptive ransac
            vector<Eigen::Vector2d> image_pts_;
            vector<Eigen::Vector2d> camera_pan_tilt_;
            
            Hypothesis()
            {
                loss_ = INT_MAX;
            }
            
            Hypothesis(double loss)
            {
                loss_  = loss;
            }
            
            Hypothesis(const Hypothesis& other)
            {
                if (&other == this) {
                    return;
                }
                loss_ = other.loss_;
                ptz_ = other.ptz_;
                inlier_indices_ = other.inlier_indices_;
                inlier_candidate_pan_tilt_indices_ = other.inlier_candidate_pan_tilt_indices_;
                image_pts_ = other.image_pts_;
                camera_pan_tilt_ = other.camera_pan_tilt_;
            }
            
            Hypothesis & operator = (const Hypothesis & other)
            {
                if (&other == this) {
                    return *this;
                }
                loss_ = other.loss_;
                ptz_ = other.ptz_;
                inlier_indices_ = other.inlier_indices_;
                inlier_candidate_pan_tilt_indices_ = other.inlier_candidate_pan_tilt_indices_;
                image_pts_ = other.image_pts_;
                camera_pan_tilt_ = other.camera_pan_tilt_;
                
                return *this;
            }
            
            bool operator < (const Hypothesis & other) const
            {
                return loss_ < other.loss_;
            }
        };
    }
    
    // @brief project pan, tilt ray to image space
    static vector<vector<Eigen::Vector2d> > projectPanTilt(const Eigen::Vector3d& ptz,
                                                           const Eigen::Vector2d& pp,
                                                           const vector<vector<Eigen::Vector2d> > & input_pan_tilt)
    {
        vector<vector<Eigen::Vector2d> > image_pts(input_pan_tilt.size());
        for (int i = 0; i<input_pan_tilt.size(); i++) {
            for (int j = 0; j<input_pan_tilt[i].size(); j++) {
                Eigen::Vector2d point = cvx_pgl::panTilt2Point(pp, ptz, input_pan_tilt[i][j]);
                image_pts[i].push_back(point);
            }
        }
        return image_pts;
    }
    
    bool preemptiveRANSACOneToMany(const vector<Eigen::Vector2d> & image_points,
                                   const vector<vector<Eigen::Vector2d> > & candidate_pan_tilt,
                                   const Eigen::Vector2d& pp,
                                   const PTZPreemptiveRANSACParameter & param,
                                   Eigen::Vector3d & ptz,
                                   bool verbose)
    {
        assert(image_points.size() == candidate_pan_tilt.size());
        if (image_points.size() <= 12) {
            return false;
        }
        
        const int num_iteration = 1024;
        const int K = 512;
        const int N = (int)image_points.size();
        const int B = param.sample_number_;
        double threshold = param.reprojection_error_threshold_;
        
        // step 1: sample hyperthesis
        vector<Hypothesis> hypotheses;
        for (int i = 0; i<num_iteration; i++) {
            int k1 = 0;
            int k2 = 0;
            do{
                k1 = rand()%N;
                k2 = rand()%N;
            }while (k1 == k2);
            
            const Eigen::Vector2d pan_tilt1 = candidate_pan_tilt[k1][0];
            const Eigen::Vector2d pan_tilt2 = candidate_pan_tilt[k2][0];
            const Eigen::Vector2d point1 = image_points[k1];
            const Eigen::Vector2d point2 = image_points[k2];
            Eigen::Vector3d ptz;            
            
            
            bool is_valid = EigenX::ptzFromTwoPoints(pan_tilt1, pan_tilt2, point1, point2, pp, ptz);
            if (is_valid) {
                Hypothesis hp;
                hp.ptz_ = ptz;
                hypotheses.push_back(hp);
            }
            else {
                if (verbose) {
                    printf("warning: estimate ptz from two points failed.\n");
                }
                
            }
            if (hypotheses.size() > K) {
                if (verbose) {
                    printf("initialization repeat %d times\n", i);
                }
                break;
            }
        }
        if (verbose) {
            printf("init ptz camera parameter number is %lu\n", hypotheses.size());
        }
        
        if (hypotheses.size() < K/4) {
            printf("Error: not enough hypotheses %lu vs %d.\n", hypotheses.size(), K/4);
            return false;
        }
        
        // step 2: optimize pan, tilt, focal length
        while (hypotheses.size() > 1) {
            // sample random set
            vector<Eigen::Vector2d> sampled_image_pts;
            vector<vector<Eigen::Vector2d> > sampled_pan_tilt;  // one camera point may have multiple pan, tilt correspondences
            vector<int> sampled_indices;
            for (int i =0; i<B; i++) {
                int index = rand()%N;
                sampled_image_pts.push_back(image_points[index]);
                sampled_pan_tilt.push_back(candidate_pan_tilt[index]);
                sampled_indices.push_back(index);
            }
            
            // count outliers as energy measurement
            for (int i = 0; i<hypotheses.size(); i++) {
                // check accuracy by project pan, tilt to image space
                vector<vector<Eigen::Vector2d> > projected_pan_tilt = projectPanTilt(hypotheses[i].ptz_, pp, sampled_pan_tilt);
                
                // check minimum distance from projected points to image coordinate
                for (int j = 0; j<projected_pan_tilt.size(); j++) {
                    double min_dis = threshold * 2;
                    int min_index = -1;
                    for (int k = 0; k<projected_pan_tilt[j].size(); k++) {
                        Eigen::Vector2d dif = sampled_image_pts[j] - projected_pan_tilt[j][k];
                        double dis = dif.norm();
                        if (dis < min_dis) {
                            min_dis = dis;
                            min_index = k;
                        }
                    } // end of k
                    
                    if (min_dis > threshold) {
                        hypotheses[i].loss_ += 1.0;
                    }
                    else {
                        hypotheses[i].inlier_indices_.push_back(sampled_indices[j]);
                        hypotheses[i].inlier_candidate_pan_tilt_indices_.push_back(min_index);
                    }
                } // end of j
                assert(hypotheses[i].inlier_indices_.size() == hypotheses[i].inlier_candidate_pan_tilt_indices_.size());
            } // end of i
            
            // remove half of the hypotheses
            std::sort(hypotheses.begin(), hypotheses.end());
            hypotheses.resize(hypotheses.size()/2);
            
            // refine by inliers
            for (int i = 0; i<hypotheses.size(); i++) {
                // number of inliers is larger than minimum configure
                if (hypotheses[i].inlier_indices_.size() > 4) {
                    vector<Eigen::Vector2d> inlier_image_pts;
                    vector<Eigen::Vector2d> inlier_pan_tilt;
                    for (int j = 0; j < hypotheses[i].inlier_indices_.size(); j++) {
                        int index = hypotheses[i].inlier_indices_[j];
                        int pan_tilt_index = hypotheses[i].inlier_candidate_pan_tilt_indices_[j];
                        inlier_image_pts.push_back(image_points[index]);
                        inlier_pan_tilt.push_back(candidate_pan_tilt[index][pan_tilt_index]);
                    }
                    
                    Eigen::Vector3d opt_ptz;
                    double reprojection_error = cvx_pgl::optimizePTZ(pp, inlier_pan_tilt, inlier_image_pts, hypotheses[i].ptz_, opt_ptz);
                    hypotheses[i].ptz_ = opt_ptz;
                    hypotheses[i].inlier_indices_.clear();
                    hypotheses[i].inlier_candidate_pan_tilt_indices_.clear();
                    if (hypotheses.size() == 1 && verbose) {
                        printf("hypotheses rank %lu, reprojection error %f pixels\n", hypotheses.size(), reprojection_error);
                    }
                }
            }
        }
        assert(hypotheses.size() == 1);
        
        ptz = hypotheses[0].ptz_;
        return true;
    }
}