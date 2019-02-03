//
//  vpgl_ptz_camera_optimize.cpp
//  QuadCopter
//
//  Created by jimmy on 6/29/15.
//  Copyright (c) 2015 Nowhere Planet. All rights reserved.
//

#include "vpgl_ptz_camera_optimize.h"
#include <vnl/vnl_least_squares_function.h>
#include <vgl/vgl_distance.h>
#include <vnl/vnl_inverse.h>
#include <vgl/algo/vgl_homg_operators_2d.h>
#include <vnl/algo/vnl_levenberg_marquardt.h>

// clamp v to a range
// this function is disabled
static double clamp(double v, double low, double high)
{
    assert(low < high);
    if (v < low) {
        return v;
    }
    else if( v > high)
    {
        return v;
    }
    return v;
}


static vnl_matrix_fixed<double, 3, 3> homographyFromProjectiveCamera(const vpgl_perspective_camera<double> & camera)
{
    vnl_matrix_fixed<double, 3, 3> H;
    vnl_matrix_fixed<double, 3, 4> P = camera.get_matrix();
    
    H(0, 0) = P(0, 0); H(0, 1) = P(0, 1); H(0, 2) = P(0, 3);
    H(1, 0) = P(1, 0); H(1, 1) = P(1, 1); H(1, 2) = P(1, 3);
    H(2, 0) = P(2, 0); H(2, 1) = P(2, 1); H(2, 2) = P(2, 3);
    
    return H;
}

static vgl_conic<double> projectConic(const vnl_matrix_fixed<double, 3, 3> & H, const vgl_conic<double> & conic)
{
    double a = conic.a();
    double b = conic.b();
    double c = conic.c();
    double d = conic.d();
    double e = conic.e();
    double f = conic.f();
    vnl_matrix_fixed<double, 3, 3> C;
    C(0, 0) = a;     C(0, 1) = b/2.0; C(0, 2) = d/2.0;
    C(1, 0) = b/2.0; C(1, 1) = c;     C(1, 2) = e/2.0;
    C(2, 0) = d/2.0; C(2, 1) = e/2.0; C(2, 2) = f;
    // project conic by H
    vnl_matrix_fixed<double, 3, 3> H_inv = vnl_inverse(H);
    vnl_matrix_fixed<double, 3, 3> C_proj = H_inv.transpose() * C * H_inv;
    
    // approximate a conic from 3*3 matrix
    double aa =  C_proj(0, 0);
    double bb = (C_proj(0, 1) + C_proj(1, 0));
    double cc =  C_proj(1, 1);
    double dd = (C_proj(0, 2) + C_proj(2, 0));
    double ee = (C_proj(1, 2) + C_proj(2, 1));
    double ff =  C_proj(2, 2);
    
    // project conic
    vgl_conic<double> conic_proj(aa, bb, cc, dd, ee, ff);
    return conic_proj;
}

class optimize_ptz_camera_line_conic_ICP_residual: public vnl_least_squares_function
{
protected:
    const vpgl_ptz_camera init_ptz_;
    const VpglPTZCameraOptimizeParameter opt_para_;
    const vector<vgl_point_3d<double> > wldPts_;
    const vector<vgl_point_2d<double> > imgPts_;
    const vector<vgl_line_3d_2_points<double> >  wldLines_;
    const vector<vector<vgl_point_2d<double> > >  imgLinePts_;
    const vector<vgl_conic<double> >  wldConics_;
    const vector<vector<vgl_point_2d<double> > >  imgConicPts_;
public:
    optimize_ptz_camera_line_conic_ICP_residual(const vpgl_ptz_camera & init_ptz,
                                                const VpglPTZCameraOptimizeParameter & para,
                                                const vector<vgl_point_3d<double> > & wldPts,
                                                        const vector<vgl_point_2d<double> > & imgPts,
                                                        const vector<vgl_line_3d_2_points<double> > & wldLines,
                                                        const vector<vector<vgl_point_2d<double> > > & imgLinePts,
                                                        
                                                        const vector<vgl_conic<double> >  & wldConics,
                                                        const vector<vector<vgl_point_2d<double> > >  & imgConicPts,
                                                
                                                        const int num_line_and_conic_pts):
    vnl_least_squares_function(3, (unsigned int)(wldPts.size()) * 2 + num_line_and_conic_pts, no_gradient),
    init_ptz_(init_ptz),
    opt_para_(para),
    wldPts_(wldPts),
    imgPts_(imgPts),
    wldLines_(wldLines),
    imgLinePts_(imgLinePts),
    wldConics_(wldConics),
    imgConicPts_(imgConicPts)
    {
        assert(wldPts.size() == imgPts.size());
        assert(wldPts.size() >= 2);
        assert(wldLines.size() == imgLinePts.size());
        assert(wldConics.size() == imgConicPts.size());
        assert(para.outlier_threshold_ > 0);
    }
    
    void f(vnl_vector<double> const &x, vnl_vector<double> &fx)
    {
        //focal length, Rxyz, Camera_center_xyz
        double pan  = x[0];
        double tilt = x[1];
        double fl   = x[2];
        double outlier      = opt_para_.outlier_threshold_;
        double line_weight  = opt_para_.line_point_weight_;
        double conic_weight = opt_para_.conic_point_weight_;
        
        vpgl_ptz_camera ptzCamera = init_ptz_;
        bool isPTZ = ptzCamera.setPTZ(pan, tilt, fl);
        if (!isPTZ) {
            printf("PTZ optimization error.\n");
            ptzCamera = init_ptz_;
        }
        //loop all points
        int idx = 0;
        for (int i = 0; i<wldPts_.size(); i++) {
            vgl_point_2d<double> proj_p = ptzCamera.project(wldPts_[i]);
            fx[idx] = clamp(imgPts_[i].x() - proj_p.x(), -outlier, outlier);
            idx++;
            fx[idx] = clamp(imgPts_[i].y() - proj_p.y(), -outlier, outlier);
            idx++;
        }
        
        // for points locate on the line
        for (int i = 0; i<wldLines_.size(); i++) {
            vgl_point_2d<double> p1 = ptzCamera.project(wldLines_[i].point1());
            vgl_point_2d<double> p2 = ptzCamera.project(wldLines_[i].point2());
            vgl_line_2d<double> line(p1, p2);
            for (int j = 0; j<imgLinePts_[i].size(); j++) {
                vgl_point_2d<double> p3 = imgLinePts_[i][j];
                fx[idx] = clamp(vgl_distance(line, p3), -outlier, outlier) * line_weight; // clamp to outlier distance and multiple weight
                idx++;
            }
        }
        
        // for points locate on the conics
        vnl_matrix_fixed<double, 3, 3> H = ::homographyFromProjectiveCamera(ptzCamera);
        for (int i = 0; i<wldConics_.size(); i++) {
            vgl_conic<double> conic_proj = projectConic(H, wldConics_[i]);
            for (int j = 0; j<imgConicPts_[i].size(); j++) {
                vgl_point_2d<double> p = imgConicPts_[i][j];
                double dis = vgl_homg_operators_2d<double>::distance_squared(conic_proj, vgl_homg_point_2d<double>(p.x(), p.y(), 1.0));
                dis = sqrt(dis + 0.0000001);
                dis = clamp(dis, -outlier, outlier) * conic_weight;
                fx[idx] = dis;
                idx++;
            }
        }        
    }
    
    
    
    bool getCamera(vnl_vector<double> const &x, vpgl_ptz_camera & ptzCamera)
    {
        double pan  = x[0];
        double tilt = x[1];
        double fl   = x[2];
        ptzCamera = init_ptz_;
        return ptzCamera.setPTZ(pan, tilt, fl);
    }
};



bool VpglPTZCameraOptimize::optimize_PTZ_camera_ICP(const vector<vgl_point_3d<double> > &wldPts,
                                                    const vector<vgl_point_2d<double> > &imgPts,
                                                    const vector<vgl_line_3d_2_points<double> > & wldLines,
                                                    const vector<vector<vgl_point_2d<double> > > & imgLinePts,
                                                    const vector<vgl_conic<double> > & wldConics,
                                                    const vector<vector<vgl_point_2d<double> > > & imgConicPts,
                                                    const vpgl_ptz_camera & initPTZ,
                                                    const VpglPTZCameraOptimizeParameter & para,
                                                    vpgl_ptz_camera & optimizedPTZ,
                                                    vpgl_perspective_camera<double> & camera)
{
    assert(wldPts.size() == imgPts.size());
    assert(wldPts.size() >= 2);
    assert(wldLines.size() == imgLinePts.size());
    assert(wldConics.size() == imgConicPts.size());
    
    int num_line_and_conid_pts = 0;
    for (int i = 0; i<imgLinePts.size(); i++) {
        num_line_and_conid_pts += imgLinePts[i].size();
    }
    for (int i = 0; i<imgConicPts.size(); i++) {
        num_line_and_conid_pts += imgConicPts[i].size();
    }
    
    optimize_ptz_camera_line_conic_ICP_residual residual(initPTZ, para, wldPts, imgPts, wldLines, imgLinePts, wldConics, imgConicPts,
                                                                 num_line_and_conid_pts);
    vnl_vector<double> x(3, 0);
    x[0] = initPTZ.pan();
    x[1] = initPTZ.tilt();
    x[2] = initPTZ.focal_length();
    
    std::cout<<"init ptz is "<<x<<std::endl;
    vnl_levenberg_marquardt lmq(residual);
    bool isMinimied = lmq.minimize(x);
    if (!isMinimied) {
        std::cerr<<"Error: ptz camera optimize not converge.\n";
        lmq.diagnose_outcome();
        return false;
    }
    lmq.diagnose_outcome();
    residual.getCamera(x, optimizedPTZ);
    std::cout<<"optimized ptz is "<<x<<std::endl;
    
    // redundant code
    camera.set_calibration(optimizedPTZ.get_calibration());
    camera.set_camera_center(optimizedPTZ.get_camera_center());
    camera.set_rotation(optimizedPTZ.get_rotation());
    return true;
}
