#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "annotation_window.h"

void callback(cv::Point p){
	printf("%d, %d", p.x, p.y);
}

 
int main(int argc, char *argv[])
{
	cv::Mat source_img = cv::imread("test_img.jpg", cv::IMREAD_COLOR);

	AnnotationWindow window(callback);

	window.set_source_img(source_img);

	window.StartLoop();
	return 0;
}