#include "NHL_court.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

NHLCourt::NHLCourt(const char *court_file)
{
    // vcl_vector<vgl_line_segment_2d< double > > markings_;
    
    FILE *pf = fopen(court_file, "r");
    if (!pf) {
        printf("Error: can not open file %s\n", court_file);
        assert(0);
    }
    else {
        int n1 = 0;
        int n2 = 0;
        int ret_num = fscanf(pf, "%d %d", &n1, &n2);
        assert(ret_num == 2);
        vector<vgl_point_2d<double>> pts;
        for (int i = 0; i<n1; i++) {
            double x = 0.0;
            double y = 0.0;
            ret_num = fscanf(pf, "%lf %lf", &x, &y);
            assert(ret_num == 2);
            pts.push_back(vgl_point_2d<double>(x, y));
        }
        for (int i = 0; i<n2; i++) {
            int idx1 = 0;
            int idx2 = 0;
            ret_num = fscanf(pf, "%d %d", &idx1, &idx2);
            assert(ret_num == 2);
            markings_.push_back(vgl_line_segment_2d<double>(pts[idx1], pts[idx2]));
        }
        fclose(pf);
        printf("read from %s, marking number %lu.\n", court_file, markings_.size());
    }
}


void NHLCourt::overlayLines(const vpgl_perspective_camera<double> & camera,
                            Mat & image) const
{
    assert(image.channels() == 3);
    
    for ( unsigned int i = 0; i < markings_.size(); ++i )
    {
        vgl_homg_point_3d< double > p1( markings_[i].point1().x(), markings_[i].point1().y(), 0, 1.0 );
        vgl_homg_point_3d< double > p2( markings_[i].point2().x(), markings_[i].point2().y(), 0, 1.0 );
        
        if (camera.is_behind_camera(p1) || camera.is_behind_camera(p2)) {
            continue;
        }
        
        vgl_point_2d< double > start = vgl_point_2d< double >(camera.project(p1));
        vgl_point_2d< double > stop = vgl_point_2d< double >(camera.project(p2));
        
        cv::line(image, cv::Point(start.x(), start.y()),
                 cv::Point(stop.x(), stop.y()),
                 cv::Scalar(255, 0, 0));
        
        //vicl_overlay_line_segment(image, vgl_line_segment_2d< double >( start, stop ), vicl_colour::blue, 2);
    }
}

void NHLCourt::overlayPoints(const vpgl_perspective_camera<double> & camera,
                            Mat & image) const
{
    //@todo
    
}
