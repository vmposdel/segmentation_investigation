#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdio.h>
#include <iostream>

void calcHistogram(const cv::Mat& _in_image);
void calcBackProjection( const std::vector<cv::Mat>& _in_images );
void detectEdges( cv::Mat& _in_image );
void detectBlobs( cv::Mat& _in_image );

