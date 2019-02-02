

#include "play_field.h"
#include <math.h>
#include <vgl/vgl_conic.h>
#include "length_unit.h"

bool PlayField::find_candinate_point(const vgl_point_2d<double> &inPt,
                                     vgl_point_2d<double> &outPt,
                                     double threshold)
{
    vector<vgl_point_2d<double> > pts = this->candindate_points();
    //find nearest neighbor in image space
    double dis_min = INT_MAX;
    int idx_min = 0;
    for (int i = 0; i<pts.size(); i++) {
        vgl_point_2d<double> q = this->world_point_to_image_point(pts[i]);
        
        double dif_x = q.x() - inPt.x();
        double dif_y = q.y() - inPt.y();
        double dis = dif_x * dif_x + dif_y * dif_y;
        if (dis < dis_min) {
            dis_min = dis;
            idx_min = i;
        //    printf("input %f %f  output  %f %f world in feet %f %f\n", inPt.x(), inPt.y(), q.x(), q.y(), pts[i].x(), pts[i].y());
        }
    }
    //compare with threshold
    dis_min = sqrt(dis_min);
    if (dis_min <= threshold) {
        //outPt = basketball_court::world_point_to_image_point(candinate_points_[idx_min]);
        outPt = pts[idx_min];
        //printf("input %f %f  output  %f %f", inPt.x(), inPt.x(), outPt.x(), outPt.x());
      //  printf("find matching, min distance is %f\n", dis_min);
        return true;
    }
    else
    {
     //   printf("Can't find matching, min distance is %f\n", dis_min);
        return false;
    }
    
    
}
bool PlayField::find_line_intersection_candindate_point(const vgl_point_2d<double> &inPt,
                                                        vgl_point_2d<double> &outPt,
                                                        double threshold)
{
    vector<vgl_point_2d<double> > pts = this->line_intersection_candindate_points();
    double dis_min = INT_MAX;
    int idx_min = 0;
    for (int i = 0; i<pts.size(); i++) {
        vgl_point_2d<double> q = this->world_point_to_image_point(pts[i]);
        
        double dif_x = q.x() - inPt.x();
        double dif_y = q.y() - inPt.y();
        double dis = dif_x * dif_x + dif_y * dif_y;
        if (dis < dis_min) {
            dis_min = dis;
            idx_min = i;
        }
    }
    //compare with threshold
    dis_min = sqrt(dis_min);
    if (dis_min <= threshold) {
        outPt = pts[idx_min];
        return true;
    }
    else
    {
        return false;
    }
}


/***********************************************  BasketballPlayField  **********************************************/

vector<vgl_point_2d<double> > BasketballPlayField::candindate_points()
{
    vector< vgl_point_2d<double> > pts;
    
    pts.resize(26);
    
    //out line
    pts[0].set(0, 0);
    pts[1].set(94, 0);
    pts[2].set(94, 50);
    pts[3].set(0, 50);
    
    //left small poly
    pts[4].set(0, 19);
    pts[5].set(19, 19);
    pts[6].set(19, 31);
    pts[7].set(0, 31);
    
    //right small poly
    pts[8].set(75, 19);
    pts[9].set(94, 19);
    pts[10].set(94, 31);
    pts[11].set(75, 31);
    
    //point on the arc
    pts[12].set(25, 25);
    pts[13].set(69, 25);
    
    //center point on the boundary and two --|--
    pts[14].set(47, 50);
    pts[15].set(47, 0);
    
    pts[16].set(28, 50);
    pts[17].set(66, 50);
    
    //center circle point
    pts[18].set(47, 19);
    pts[19].set(47, 31);
    
   
    
    //professional 3 point line
    pts[20].set(14, 3);
    pts[21].set(94 - 14, 3);
    
    pts[22].set(11, 19);
    pts[23].set(94 - 11, 19);
    
    pts[24].set(0, 51/12.0);
    pts[25].set(94, 51/12.0);
    
    return pts;    
}
vector<vgl_point_2d<double> > BasketballPlayField::line_intersection_candindate_points()
{
    vector< vgl_point_2d<double> > pts;
    
    pts.resize(16);
    
    // two virtual --|--
    pts[0].set(28, 0);
    pts[1].set(66, 0);
    
    // extend from penalty line
    pts[2].set(19, 0);
    pts[3].set(75, 0);
    
    pts[4].set(19, 50);
    pts[5].set(75, 50);
    
    // from professional 3 point line
    pts[6].set(14, 3);
    pts[7].set(94- 14, 3);
    
    pts[8].set(14, 50-3);
    pts[9].set(94- 14, 50-3);
    
    // 3 point line with side line
    pts[10].set(0, 3);
    pts[11].set(0, 50-3);
    pts[12].set(94, 3);
    pts[13].set(94, 50-3);
    
    // logo with division line
    pts[14].set(47, 35.6203);
    pts[15].set(47, 14.32);
    
    return pts;
}

vgl_point_2d<double> BasketballPlayField::image_point_to_world_point(const vgl_point_2d<double> &p)
{
  //  vgl_point_2d<double> q;
    double x = p.x();
    double y = p.y();
    
    x = x - 70;
    y = y - 70;
    
    y = 600 - y;
    
    x = x/12;
    y = y/12;
    
    return vgl_point_2d<double>(x, y);
}

vgl_point_2d<double> BasketballPlayField::world_point_to_image_point(const vgl_point_2d<double> &p)
{
    double x = p.x();
    double y = p.y();
    
    //feet turn to inch (pixel)
    x = x * 12;
    y = y * 12;
    
    y = 600 - y;  //flip y
    
    //add 70
    x = x + 70;
    y = y + 70;
    
    return vgl_point_2d<double>(x, y);
}

/*******************          USHSBasketballPlayField         ******************/

USHSBasketballPlayField::USHSBasketballPlayField()
{
    
}
USHSBasketballPlayField::~USHSBasketballPlayField()
{
    
}

vector<vgl_point_2d<double> > USHSBasketballPlayField::candindate_points()
{
    const int pts_num = 30;
    const double width = 84.0;
    const double height = 50.0;
    
    vector<vgl_point_2d<double> > marking_pts;
    marking_pts.resize(pts_num);
    
    marking_pts[0] = vgl_point_2d<double>(0, 0);
    marking_pts[1] = vgl_point_2d<double>(width, 0);
    marking_pts[2] = vgl_point_2d<double>(width, height);
    marking_pts[3] = vgl_point_2d<double>(0, height);
    marking_pts[4] = vgl_point_2d<double>(width/2, 0);
    marking_pts[5] = vgl_point_2d<double>(width/2, height);
    
    // center circle
    marking_pts[6] = vgl_point_2d<double>(width/2, height/2 - 12/2.0);
    marking_pts[7] = vgl_point_2d<double>(width/2, height/2 + 12/2.0);
    
    // left penalty line
    marking_pts[8] = vgl_point_2d<double>(19, (height-12)/2.0);
    marking_pts[9] = vgl_point_2d<double>(19, (height-12)/2.0 + 12.0);
    
    // right penalty line
    marking_pts[10] = vgl_point_2d<double>(width - 19.0, (height-12)/2.0);
    marking_pts[11] = vgl_point_2d<double>(width - 19.0, (height-12)/2.0 + 12.0);
    
    //
    marking_pts[12] = vgl_point_2d<double>(0, (height-12)/2.0);
    marking_pts[13] = vgl_point_2d<double>(0, (height-12)/2.0 + 12.0);
    marking_pts[14] = vgl_point_2d<double>(width, (height-12)/2.0);
    marking_pts[15] = vgl_point_2d<double>(width, (height-12)/2.0 + 12.0);
    
    // virtual points
    marking_pts[16] = vgl_point_2d<double>(19, 0);
    marking_pts[17] = vgl_point_2d<double>(19, height);
    marking_pts[18] = vgl_point_2d<double>(width - 19, 0);
    marking_pts[19] = vgl_point_2d<double>(width - 19, height);
    
    // 3 point line extension, left
    const double radius_three_point = 19+6-63.0/12.0;
    const double dis_to_sideline = (height-2*radius_three_point)/2.0;
    marking_pts[20] = vgl_point_2d<double>(0, dis_to_sideline);
    marking_pts[21] = vgl_point_2d<double>(0, height - dis_to_sideline);
    marking_pts[22] = vgl_point_2d<double>(63.0/12.0, dis_to_sideline);
    marking_pts[23] = vgl_point_2d<double>(63.0/12.0, height - dis_to_sideline);
    
    // 3 point line extension, right
    marking_pts[24] = vgl_point_2d<double>(width, dis_to_sideline);
    marking_pts[25] = vgl_point_2d<double>(width, height - dis_to_sideline);
    marking_pts[26] = vgl_point_2d<double>(width - 63.0/12.0, dis_to_sideline);
    marking_pts[27] = vgl_point_2d<double>(width - 63.0/12.0, height - dis_to_sideline);
    
    // penalty circle and 3 point line intersection
    marking_pts[28] = vgl_point_2d<double>(25.0, height/2.0);
    marking_pts[29] = vgl_point_2d<double>(width - 25.0, height/2.0);
    
    return marking_pts;
}

vector<vgl_point_2d<double> > USHSBasketballPlayField::line_intersection_candindate_points()
{
    const double width = 84.0;
    const double height = 50.0;
    vector<vgl_point_2d<double> > pts;
    pts.resize(4);
    // virtual points
    pts[0] = vgl_point_2d<double>(19, 0);
    pts[1] = vgl_point_2d<double>(19, height);
    pts[2] = vgl_point_2d<double>(width - 19, 0);
    pts[3] = vgl_point_2d<double>(width - 19, height);
    return pts;
}

// exchange between pixel and world coordinate. This function depends on court image
vgl_point_2d<double> USHSBasketballPlayField::image_point_to_world_point(const vgl_point_2d<double> &p)
{
    double x = p.x();
    double y = p.y();
    
    x = x - 70;
    y = y - 70;
    
    y = 600 - y;
    
    x = x/12.0;
    y = y/12.0;
    
    return vgl_point_2d<double>(x, y);
}

vgl_point_2d<double> USHSBasketballPlayField::world_point_to_image_point(const vgl_point_2d<double> &p)
{
    double x = p.x();
    double y = p.y();
    
    //feet turn to inch (pixel)
    x = x * 12;
    y = y * 12;
    
    y = 600 - y;  //flip y
    
    //add 70
    x = x + 70;
    y = y + 70;
    
    return vgl_point_2d<double>(x, y);
}

/**********************************************  WWosSoccerPlayField  ***********************************************/

vector<vgl_point_2d<double> > WWosSoccerPlayField::candindate_points()
{
    double width = 118;
    double height = 70;
    
    
    vector<vgl_point_2d<double> > pts(43);
    
    // left goal line
    pts[0].set(0, 0);
    pts[1].set(0, 10);
    pts[2].set(0, height/2 - 4);
    pts[3].set(0, height/2 - 10);
    pts[4].set(0, height/2 - 22);
    pts[5].set(0, height/2 + 4);
    pts[6].set(0, height/2 + 10);
    pts[7].set(0, height/2 + 22);
    pts[8].set(0, height - 10);
    pts[9].set(0, height);
    
    // 6 yard
    pts[10].set(6, height/2 - 10);
    pts[11].set(6, height/2 + 10);
    
    // 10 yard
    pts[12].set(10, 0);
    pts[13].set(10, height);
    
    // 18 yard
    pts[14].set(18, height/2 - 22);
    pts[15].set(18, height/2 + 22);
    pts[16].set(18, height/2 - 8);
    pts[17].set(18, height/2 + 8);
    
    // center line
    pts[18].set(width/2, 0);
    pts[19].set(width/2, height);
    pts[20].set(width/2, height/2);
    pts[21].set(width/2, height/2 - 10);
    pts[22].set(width/2, height/2 + 10);
    
    // right side
    // 18 yard
    pts[23].set(width - 18, height/2 - 22);
    pts[24].set(width - 18, height/2 + 22);
    pts[25].set(width - 18, height/2 - 8);
    pts[26].set(width - 18, height/2 + 8);
    
    // 10 yard
    pts[27].set(width - 10, 0);
    pts[28].set(width - 10, height);
    
    // 6 yard
    pts[29].set(width - 6, height/2 - 10);
    pts[30].set(width - 6, height/2 + 10);
    
    pts[31].set(width, 0);
    pts[32].set(width, 10);
    pts[33].set(width, height/2 - 4);
    pts[34].set(width, height/2 - 10);
    pts[35].set(width, height/2 - 22);
    pts[36].set(width, height/2 + 4);
    pts[37].set(width, height/2 + 10);
    pts[38].set(width, height/2 + 22);
    pts[39].set(width, height - 10);
    pts[40].set(width, height);
    
    // penelty point
    pts[41].set(12, height/2);
    pts[42].set(width-12, height/2);
    
    
    // yard to feet
    for (int i = 0; i<pts.size(); i++) {
        pts[i].set(pts[i].x() * 3.0, pts[i].y() * 3.0);
    }
    return pts;

    
}
vector<vgl_point_2d<double> > WWosSoccerPlayField::line_intersection_candindate_points()
{
    
    double width = 118;
    double height = 70;
    vector<vgl_point_2d<double> > pts;
    
    // two points in the center circle
    pts.push_back(vgl_point_2d<double>(width/2 - 10, height/2));
    pts.push_back(vgl_point_2d<double>(width/2 + 10, height/2));
    
    // align with the two black line
    pts.push_back(vgl_point_2d<double>(width/2 - 10, height));
    pts.push_back(vgl_point_2d<double>(width/2 + 10, height));
    
    // intersection with 18 yard line
    pts.push_back(vgl_point_2d<double>(18, 0));
    pts.push_back(vgl_point_2d<double>(width - 18, 0));
    
    pts.push_back(vgl_point_2d<double>(18, height));
    pts.push_back(vgl_point_2d<double>(width - 18, height));
    
    // field hockey points
    pts.push_back(vgl_point_2d<double>(width/2 - 10, 11.6));
    pts.push_back(vgl_point_2d<double>(width/2 + 10, 11.6));
    
    // yard to feet
    for (int i = 0; i<pts.size(); i++) {
        pts[i].set(pts[i].x() * 3.0, pts[i].y() * 3.0);
    }
     
    
    return pts;
}
     
vgl_point_2d<double> WWosSoccerPlayField::image_point_to_world_point(const vgl_point_2d<double> &p)
{
    double x = p.x();
    double y = p.y();
    
    x = x - 70;
    y = y - 70;
    
    y = 840 - y;
    
    x = x * 3 / 12;
    y = y * 3 / 12;
    
    return vgl_point_2d<double>(x, y);
}
vgl_point_2d<double> WWosSoccerPlayField::world_point_to_image_point(const vgl_point_2d<double> &p)
{
    // feet to inch, every 3 inch as a pixel
    double x = p.x() * 12 /3;
    double y = p.y() * 12 /3;
    
    // flip y
    y = 840 - y;   // 1 yard is 3 feet
    
    x += 70;
    y += 70;
    
    return vgl_point_2d<double>(x, y);
}

vgl_conic<double> WWosSoccerPlayField::getCenterCircle(void)
{
    double width = 118;
    double height = 70;
    const double x = width * LU_YARD2METER * 0.5;
    const double y = height * LU_YARD2METER * 0.5;
    const double radius = 10.0 * LU_YARD2METER;
    vgl_conic<double > conic(vgl_homg_point_2d<double>(x, y, 1.0), radius, radius, 0.0);
    return conic;
}


/****************   Worldcup2014PlayField: public PlayField      **********************/
vector<vgl_point_2d<double> > Worldcup2014PlayField::candindate_points()
{
    double width = 115;
    double height = 74;
    
    vector<vgl_point_2d<double> > pts(43);
    
    // left goal line
    pts[0].set(0, 0);
    pts[1].set(0, 10);
    pts[2].set(0, height/2 - 4);
    pts[3].set(0, height/2 - 10);
    pts[4].set(0, height/2 - 22);
    pts[5].set(0, height/2 + 4);
    pts[6].set(0, height/2 + 10);
    pts[7].set(0, height/2 + 22);
    pts[8].set(0, height - 10);
    pts[9].set(0, height);
    
    // 6 yard
    pts[10].set(6, height/2 - 10);
    pts[11].set(6, height/2 + 10);
    
    // 10 yard
    pts[12].set(10, 0);
    pts[13].set(10, height);
    
    // 18 yard
    pts[14].set(18, height/2 - 22);
    pts[15].set(18, height/2 + 22);
    pts[16].set(18, height/2 - 8);
    pts[17].set(18, height/2 + 8);
    
    // center line
    pts[18].set(width/2, 0);
    pts[19].set(width/2, height);
    pts[20].set(width/2, height/2);
    pts[21].set(width/2, height/2 - 10);
    pts[22].set(width/2, height/2 + 10);
    
    // right side
    // 18 yard
    pts[23].set(width - 18, height/2 - 22);
    pts[24].set(width - 18, height/2 + 22);
    pts[25].set(width - 18, height/2 - 8);
    pts[26].set(width - 18, height/2 + 8);
    
    // 10 yard
    pts[27].set(width - 10, 0);
    pts[28].set(width - 10, height);
    
    // 6 yard
    pts[29].set(width - 6, height/2 - 10);
    pts[30].set(width - 6, height/2 + 10);
    
    pts[31].set(width, 0);
    pts[32].set(width, 10);
    pts[33].set(width, height/2 - 4);
    pts[34].set(width, height/2 - 10);
    pts[35].set(width, height/2 - 22);
    pts[36].set(width, height/2 + 4);
    pts[37].set(width, height/2 + 10);
    pts[38].set(width, height/2 + 22);
    pts[39].set(width, height - 10);
    pts[40].set(width, height);
    
    // penelty point
    pts[41].set(12, height/2);
    pts[42].set(width-12, height/2);
    
    
    // yard to feet
    for (int i = 0; i<pts.size(); i++) {
        pts[i].set(pts[i].x() * 3.0, pts[i].y() * 3.0);
    }
    return pts;
    
    
}
vector<vgl_point_2d<double> > Worldcup2014PlayField::line_intersection_candindate_points()
{
    
    double width = 115;
    double height = 74;
    vector<vgl_point_2d<double> > pts;
    
    // two points in the center circle
    pts.push_back(vgl_point_2d<double>(width/2 - 10, height/2));
    pts.push_back(vgl_point_2d<double>(width/2 + 10, height/2));
    
    // align with the two black line
    pts.push_back(vgl_point_2d<double>(width/2 - 10, height));
    pts.push_back(vgl_point_2d<double>(width/2 + 10, height));
    
    // intersection with 18 yard line
    pts.push_back(vgl_point_2d<double>(18, 0));
    pts.push_back(vgl_point_2d<double>(width - 18, 0));
    
    pts.push_back(vgl_point_2d<double>(18, height));
    pts.push_back(vgl_point_2d<double>(width - 18, height));
    
    
    // yard to feet
    for (int i = 0; i<pts.size(); i++) {
        pts[i].set(pts[i].x() * 3.0, pts[i].y() * 3.0);
    }
    
    
    return pts;
}

vgl_point_2d<double> Worldcup2014PlayField::image_point_to_world_point(const vgl_point_2d<double> &p)
{
    double x = p.x();
    double y = p.y();
    
    x = x - 70;
    y = y - 70;
    
    y = 888 - y;  // 74 * 12
    
    x = x * 3 / 12;
    y = y * 3 / 12;
    
    return vgl_point_2d<double>(x, y);
}
vgl_point_2d<double> Worldcup2014PlayField::world_point_to_image_point(const vgl_point_2d<double> &p)
{
    // feet to inch, every 3 inch as a pixel
    double x = p.x() * 12 /3;
    double y = p.y() * 12 /3;
    
    // flip y
    y = 888 - y; // 74 * 12, one yard is 3 feet
    
    x += 70;
    y += 70;
    
    return vgl_point_2d<double>(x, y);
}

vgl_conic<double> Worldcup2014PlayField::getCenterCircle(void)
{
    double width = 115;
    double height = 74;
    const double x = width * LU_YARD2METER * 0.5;
    const double y = height * LU_YARD2METER * 0.5;
    const double radius = 10.0 * LU_YARD2METER;
    vgl_conic<double > conic(vgl_homg_point_2d<double>(x, y, 1.0), radius, radius, 0.0);
    return conic;
}

/****************************************** VolleyballPlayField ****************************/
vector<vgl_point_2d<double> > VolleyballPlayField::candindate_points()
{
    const double width = 18;
    const double height = 9;
    
    vector<vgl_point_2d<double> > pts(10);
    pts[0].set(0, 0);
    pts[1].set(0, height);
    
    pts[2].set(6, 0);
    pts[3].set(6, height);
    
    pts[4].set(width/2, 0);
    pts[5].set(width/2, height);
    
    pts[6].set(12, 0);
    pts[7].set(12, height);
    
    pts[8].set(18, 0);
    pts[9].set(18, height);
    return pts;
}

vector<vgl_point_2d<double> > VolleyballPlayField::line_intersection_candindate_points()
{
    
    return vector<vgl_point_2d<double> >();
    
}
vgl_point_2d<double> VolleyballPlayField::image_point_to_world_point(const vgl_point_2d<double> &p)
{
    double x = p.x();
    double y = p.y();
    
    x = x - 70;
    y = y - 70;
    
    y = 450 - y;  // 9 * 50
    
    x = x/50;
    y = y/50;
    
    return vgl_point_2d<double>(x, y);
}

vgl_point_2d<double> VolleyballPlayField::world_point_to_image_point(const vgl_point_2d<double> &p)
{
    // every 1 meter is 50 pixel
    double x = p.x() * 50;
    double y = p.y() * 50;
    
    // flip y
    y = 450 - y; // 9 * 50
    
    x += 70;
    y += 70;
    
    return vgl_point_2d<double>(x, y);
}


/****************************************** NHLIceHockeyPlayField ****************************/

vector<vgl_point_2d<double> > NHLIceHockeyPlayField::candindate_points()
{
    vector<vgl_point_2d<double> > pts(40);
    double w = 200;
    double h = 85;
    double half_w = w/2.0;
    double half_h = h/2.0;
    
    double w1 = 11;
    double w2 = 64;
    double w3 = 25;
    double w4 = 28;
    double w5 = 20;
    double w6 = 31;
    double h1 = 11;
    double h4 = 28;
    double h5 = 22;
    
    // blue line intersection
    pts[0].set(half_w - w3, 0);
    pts[1].set(half_w - w3, h);
    pts[2].set(half_w + w3, 0);
    pts[3].set(half_w + w3, h);
    
    // left goal crease
    pts[4].set(w1, half_h-4);
    pts[5].set(w1, half_h+4);
    
    // right goal crease
    pts[6].set(w-w1, half_h-4);
    pts[7].set(w-w1, half_h+4);
    
    // center line intersection
    pts[8].set(half_w, 0);
    pts[9].set(half_w, h);
    pts[10].set(half_w, half_h-15);
    pts[11].set(half_w, half_h+15);
    
    // four faceoff spot
    pts[12].set(half_w-w5, half_h+h5);
    pts[13].set(half_w-w5, half_h-h5);
    pts[14].set(half_w+w5, half_h+h5);
    pts[15].set(half_w+w5, half_h-h5);
    
    // four small circle center
    pts[16].set(w6, half_h+h5);
    pts[17].set(w6, half_h-h5);
    pts[18].set(w-w6, half_h+h5);
    pts[19].set(w-w6, half_h-h5);
    
    // four point in goal crease
    pts[20].set(w1, half_h+11);
    pts[21].set(w1, half_h-11);
    pts[22].set(w-w1, half_h+11);
    pts[23].set(w-w1, half_h-11);
    
    // 4 virtual points on the circles
    double radius = 15.0;
    pts[24].set(w6, half_h+h5+radius);
    pts[25].set(w6, half_h+h5-radius);
    
    pts[26].set(w6, half_h-h5+radius);
    pts[27].set(w6, half_h-h5-radius);
    
    pts[28].set(w-w6, half_h+h5+radius);
    pts[29].set(w-w6, half_h+h5-radius);
    
    pts[30].set(w-w6, half_h-h5+radius);
    pts[31].set(w-w6, half_h-h5-radius);
    
    // goal line and arc intersection
    double deleta_1 = 28 - 22.25;
    pts[32].set(w1, h-deleta_1);
    pts[33].set(w-w1, h-deleta_1);
    
    pts[34].set(w1, deleta_1);
    pts[35].set(w-w1, deleta_1);
    
    // goal pool
    // left goal crease
    pts[36].set(w1, half_h-3);
    pts[37].set(w1, half_h+3);
    
    // right goal crease
    pts[38].set(w-w1, half_h-3);
    pts[39].set(w-w1, half_h+3);
    return pts;
    
}

vector<vgl_point_2d<double> > NHLIceHockeyPlayField::line_intersection_candindate_points()
{    
    return NHLIceHockeyPlayField::candindate_points();
}

vgl_point_2d<double> NHLIceHockeyPlayField::image_point_to_world_point(const vgl_point_2d<double> &p)
{
    double x = p.x();
    double y = p.y();
    
    x = x - 70;
    y = y - 70;
    
    y = 85 * 12 - y;  // height 85 feet
    
    x = x * 12;
    y = y * 12;
    
    return vgl_point_2d<double>(x, y);
}

vgl_point_2d<double> NHLIceHockeyPlayField::world_point_to_image_point(const vgl_point_2d<double> &p)
{
    // feet to inch, every 1 inch as a pixel
    double x = p.x() * 12;
    double y = p.y() * 12;
    
    // flip y
    y = 85 * 12 - y; // // height 85 feet
    
    x += 70;
    y += 70;
    
    return vgl_point_2d<double>(x, y);
}

vector<vgl_conic<double>> NHLIceHockeyPlayField::getCircles(void)
{
    
    vector<vgl_conic<double>> circles;
    vector<vgl_point_2d<double>> centers;
    double radius = 15.0 * LU_FEET2METER; // feet
    double w = 200;
    double h = 85;
    double w1 = 11 + 20;
    double h1 = 22;
    centers.push_back(vgl_point_2d<double>(w/2, h/2));
    centers.push_back(vgl_point_2d<double>(w1, h/2+h1));
    centers.push_back(vgl_point_2d<double>(w1, h/2-h1));
    centers.push_back(vgl_point_2d<double>(w - w1, h/2+h1));
    centers.push_back(vgl_point_2d<double>(w - w1, h/2-h1));
    
    for (int i = 0; i<centers.size(); i++) {
        double x = centers[i].x() * LU_FEET2METER;
        double y = centers[i].y() * LU_FEET2METER;
        vgl_conic<double> conic(vgl_homg_point_2d<double>(x, y, 1.0), radius, radius, 0.0);
        circles.push_back(conic);
    }
    return circles;
}



