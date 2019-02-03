//
//  camera_estimation.cpp
//  Annotation
//
//  Created by jimmy on 2019-01-27.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "camera_estimation.hpp"
#include <vpgl/algo/vpgl_calibration_matrix_compute.h>
#include <vpgl/algo/vpgl_camera_compute.h>
#include <vnl/vnl_least_squares_function.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>
#include <vnl/vnl_inverse.h>
#include <vnl/algo/vnl_matrix_inverse.h>
#include "bcv_vgl_h_matrix_2d_compute_linear.h"
#include "bcv_vgl_h_matrix_2d_optimize_lmq.h"
#include "bcv_vgl_h_matrix_2d_decompose.h"


namespace cvx {
    bool init_calib(const vector<vgl_point_2d<double> > &wld_pts,
                                   const vector<vgl_point_2d<double> > &img_pts,
                                   const vgl_point_2d<double> &principlePoint,
                                   vpgl_perspective_camera<double> &camera)
    {
        if (wld_pts.size() < 4 && img_pts.size() < 4) {
            return false;
        }
        if (wld_pts.size() != img_pts.size()) {
            return false;
        }
        assert(wld_pts.size() >= 4 && img_pts.size() >= 4);
        assert(wld_pts.size() == img_pts.size());
        
        vpgl_calibration_matrix<double> K;
        if (vpgl_calibration_matrix_compute::natural(img_pts, wld_pts, principlePoint, K) == false) {
            std::cerr<<"Failed to compute K"<<std::endl;
            std::cerr<<"Default principle point: "<<principlePoint<<std::endl;
            return false;
        }
        
        camera.set_calibration(K);
        
        // vpgl_perspective_camera_compute_positiveZ
        if (vpgl_perspective_camera_compute::compute(img_pts, wld_pts, camera) == false) {
            std::cerr<<"Failed to computer R, C"<<std::endl;
            return false;
        }
        return true;
    }
    
    
    class optimize_perspective_camera_residual:public vnl_least_squares_function
    {
    protected:
        const std::vector<vgl_point_2d<double> > wldPts_;
        const std::vector<vgl_point_2d<double> > imgPts_;
        const vgl_point_2d<double> principlePoint_;
        
    public:
        optimize_perspective_camera_residual(const std::vector<vgl_point_2d<double> > & wldPts,
                                             const std::vector<vgl_point_2d<double> > & imgPts,
                                             const vgl_point_2d<double> & pp):
        vnl_least_squares_function(7, (unsigned int)(wldPts.size()) * 2, no_gradient),
        wldPts_(wldPts),
        imgPts_(imgPts),
        principlePoint_(pp)
        {
            assert(wldPts.size() == imgPts.size());
            assert(wldPts.size() >= 4);
        }
        
        void f(vnl_vector<double> const &x, vnl_vector<double> &fx)
        {
            //focal length, Rxyz, Camera_center_xyz
            vpgl_calibration_matrix<double> K(x[0], principlePoint_);
            
            vnl_vector_fixed<double, 3> rod(x[1], x[2], x[3]);
            vgl_rotation_3d<double>  R(rod);
            vgl_point_3d<double> cc(x[4], x[5], x[6]);  //camera center
            
            vpgl_perspective_camera<double> camera;
            camera.set_calibration(K);
            camera.set_rotation(R);
            camera.set_camera_center(cc);
            
            //loop all points
            int idx = 0;
            for (int i = 0; i<wldPts_.size(); i++) {
                vgl_point_3d<double> p(wldPts_[i].x(), wldPts_[i].y(), 0);
                vgl_point_2d<double> proj_p = (vgl_point_2d<double>)camera.project(p);
                
                fx[idx] = imgPts_[i].x() - proj_p.x();
                idx++;
                fx[idx] = imgPts_[i].y() - proj_p.y();
                idx++;
            }
        }
        
        void getCamera(vnl_vector<double> const &x, vpgl_perspective_camera<double> &camera)
        {
            
            vpgl_calibration_matrix<double> K(x[0], principlePoint_);
            
            vnl_vector_fixed<double, 3> rod(x[1], x[2], x[3]);
            vgl_rotation_3d<double>  R(rod);
            vgl_point_3d<double> camera_center(x[4], x[5], x[6]);
            
            camera.set_calibration(K);
            camera.set_rotation(R);
            camera.set_camera_center(camera_center);
        }
        
    };
    
    
    bool optimize_perspective_camera(const std::vector<vgl_point_2d<double> > & wld_pts,
                                    const std::vector<vgl_point_2d<double> > & img_pts,
                                    const vpgl_perspective_camera<double> &init_camera,
                                    vpgl_perspective_camera<double> & final_camera)
    {
        assert(wld_pts.size() == img_pts.size());
        assert(wld_pts.size() >= 4);
        
        optimize_perspective_camera_residual residual(wld_pts, img_pts,
                                                      init_camera.get_calibration().principal_point());
        
        vnl_vector<double> x(7, 0);
        x[0] = init_camera.get_calibration().get_matrix()[0][0];
        x[1] = init_camera.get_rotation().as_rodrigues()[0];
        x[2] = init_camera.get_rotation().as_rodrigues()[1];
        x[3] = init_camera.get_rotation().as_rodrigues()[2];
        x[4] = init_camera.camera_center().x();
        x[5] = init_camera.camera_center().y();
        x[6] = init_camera.camera_center().z();
        
        vnl_levenberg_marquardt lmq(residual);
        
        bool isMinimied = lmq.minimize(x);
        if (!isMinimied) {
            std::cerr<<"Error: perspective camera optimize not converge.\n";
            lmq.diagnose_outcome();
            return false;
        }
        lmq.diagnose_outcome();
        
        //    lmq.diagnose_outcome();
        residual.getCamera(x, final_camera);
        return true;
    }
    
    
    bool init_calib(const vector<vgl_point_2d<double> >& world_pts,
                              const vector<vgl_point_2d<double> >& image_pts,
                              const vector<vgl_line_segment_2d<double>>& world_line_segment,
                              const vector<vgl_line_segment_2d<double>>& image_line_segment,
                              const vgl_point_2d<double> &principle_point,
                              vpgl_perspective_camera<double> &camera)
    {
        assert(world_pts.size() == image_pts.size());
        assert(world_line_segment.size() == image_line_segment.size());
        
        // step 1: estimate H_world_to_image, First, image to world, then invert
        vector<vgl_homg_point_2d<double> > points1, points2;
        for (int i = 0; i<world_pts.size(); i++) {
            points1.push_back(vgl_homg_point_2d<double>(image_pts[i]));
            points2.push_back(vgl_homg_point_2d<double>(world_pts[i]));
        }
        
        vector<vgl_homg_line_2d<double> > lines1, lines2;
        vector<vector<vgl_homg_point_2d<double>> > point_on_line1;
        for (int i = 0; i<world_line_segment.size(); i++) {
            vgl_homg_point_2d<double> p1 = vgl_homg_point_2d<double>(image_line_segment[i].point1());
            vgl_homg_point_2d<double> p2 = vgl_homg_point_2d<double>(image_line_segment[i].point2());
            lines1.push_back(vgl_homg_line_2d<double>(p1, p2));
            
            vector<vgl_homg_point_2d<double>> pts;
            pts.push_back(p1);
            pts.push_back(p2);
            point_on_line1.push_back(pts);
            
            p1 = vgl_homg_point_2d<double>(world_line_segment[i].point1());
            p2 = vgl_homg_point_2d<double>(world_line_segment[i].point2());
            lines2.push_back(vgl_homg_line_2d<double>(p1, p2));
        }
        
        bool is_valid = false;
        bcv_vgl_h_matrix_2d_compute_linear hcl;
        vgl_h_matrix_2d<double> H;
        is_valid = hcl.compute_pl(points1, points2, lines1, lines2, H);
        if (!is_valid) {
            return false;
        }
        
        bcv_vgl_h_matrix_2d_optimize_lmq hcl_lmq(H);
        vgl_h_matrix_2d<double> opt_h;
        is_valid = hcl_lmq.optimize_pl(points1, points2, point_on_line1, lines2, opt_h);
        if (!is_valid) {
            return false;
        }
        
        // step 2: estimate K
        vpgl_calibration_matrix<double> K;
        is_valid = vpgl_calibration_matrix_compute::natural(opt_h.get_inverse(), principle_point, K);
        if (!is_valid) {
            return false;
        }
        
        
        vnl_matrix_fixed<double, 3, 3> h_world_to_img = opt_h.get_inverse().get_matrix();
        //cout<<"homography is "<<h_world_to_img<<endl;
        bcv_vgl_h_matrix_2d_decompose hd;
        std::vector<vnl_matrix_fixed<double, 3, 3>> rotations;
        std::vector<vnl_vector_fixed<double, 3>> translations;
        is_valid = hd.compute(K.get_matrix(), h_world_to_img.as_matrix(), rotations, translations);
        if (!is_valid) {
            return false;
        }
        
        assert(rotations.size() == translations.size());
        assert(rotations.size() == 2);
        
        vnl_matrix<double> invR1 = vnl_matrix_inverse<double>(rotations[0].as_matrix());
        vnl_vector<double> cc1 = -invR1*translations[0];
        
        vnl_matrix<double> invR2 = vnl_matrix_inverse<double>(rotations[1].as_matrix());
        vnl_vector<double> cc2 = -invR2*translations[1];
        
        if (cc1[2] < 0 && cc2[2] < 0) {
            printf("Warning: two solutions are below z = 0 plane \n");
            return false;
        }
        else if(cc1[2] >= 0 && cc2[2] >= 0) {
            printf("Warning: two ambiguity solutions \n");
            return false;
        }
        
        vnl_matrix_fixed<double, 3, 3> R = cc1[2]> 0? rotations[0]:rotations[1];
        vnl_vector<double> cc   = cc1[2]>0 ?cc1:cc2;
        
        camera.set_calibration(K);
        camera.set_rotation(vgl_rotation_3d<double>(R));
        camera.set_camera_center(vgl_point_3d<double>(cc[0], cc[1], cc[2]));
        
        return true;
    }


    
    
}









