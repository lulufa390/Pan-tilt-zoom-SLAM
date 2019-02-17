//
//  io_util.cpp
//  CameraCalibration
//
//  Created by jimmy on 2019-02-16.
//  Copyright © 2019 Nowhere Planet. All rights reserved.
//

#include "io_util.hpp"
#include <iostream>

using std::string;
using std::cout;

namespace io_util {
    bool writeCamera(const char *file_name, const char *image_name, const vpgl_perspective_camera<double> & camera)
    {
        assert(file_name);
        assert(image_name);
        
        FILE *pf = fopen(file_name, "w");
        if (!pf) {
            printf("can not create file %s\n", file_name);
            return false;
        }
        fprintf(pf, "%s\n", image_name);
        fprintf(pf, "ppx\t ppy\t focal length\t Rx\t Ry\t Rz\t Cx\t Cy\t Cz\n");
        double ppx = camera.get_calibration().principal_point().x();
        double ppy = camera.get_calibration().principal_point().y();
        double fl = camera.get_calibration().get_matrix()[0][0];
        double Rx = camera.get_rotation().as_rodrigues()[0];
        double Ry = camera.get_rotation().as_rodrigues()[1];
        double Rz = camera.get_rotation().as_rodrigues()[2];
        double Cx = camera.get_camera_center().x();
        double Cy = camera.get_camera_center().y();
        double Cz = camera.get_camera_center().z();
        fprintf(pf, "%f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\t %f\n", ppx, ppy, fl, Rx, Ry, Rz, Cx, Cy, Cz);
        fclose(pf);
        return true;
    }
    
    bool readCamera(const char *file_name, string & image_name, vpgl_perspective_camera<double> & camera)
    {
        assert(file_name);
        FILE *pf = fopen(file_name, "r");
        if (!pf) {
            printf("can not open file %s\n", file_name);
            return false;
        }
        char buf[1024] = {NULL};
        int num = fscanf(pf, "%s\n", buf);
        assert(num == 1);
        image_name = string(buf);
        for (int i = 0; i<1; i++) {
            char lineBuf[BUFSIZ] = {NULL};
            fgets(lineBuf, sizeof(lineBuf), pf);
            cout<<lineBuf;
        }
        double ppx, ppy, fl, rx, ry, rz, cx, cy, cz;
        int ret = fscanf(pf, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &ppx, &ppy, &fl, &rx, &ry, &rz, &cx, &cy, &cz);
        if (ret != 9) {
            printf("Error: read camera parameters!\n");
            return false;
        }
        
        vpgl_calibration_matrix<double> K(fl, vgl_point_2d<double>(ppx, ppy));
        vnl_vector_fixed<double, 3> rod(rx, ry, rz);
        vgl_rotation_3d<double> R(rod);
        vgl_point_3d<double> cc(cx, cy, cz);
        
        camera.set_calibration(K);
        camera.set_rotation(R);
        camera.set_camera_center(cc);
        fclose(pf);
        
        return true;
    }
}
