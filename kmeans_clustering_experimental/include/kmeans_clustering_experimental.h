#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/core/core.hpp"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <sys/time.h>
 

int cameraId;
int maxClusters;
//cv::Scalar colorTab[2]; 
//int* colors;
cv::Vec3b* colors;

void initParams();

static void captureFrame(std::vector<cv::Mat>& inImages, std::vector<std::string>& imageNames);

void calcHistogram(cv::Mat& frame, cv::Mat& histNorm);

void calcClustered(cv::Mat& frame, cv::Mat& kmeansFrame);

void detectEdges(cv::Mat& frame, cv::Mat& edgedFrame);

void showImages(cv::Mat& frame, cv::Mat& kmeansFrame, cv::Mat& edgedFrame);
