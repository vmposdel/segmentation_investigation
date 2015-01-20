#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdio.h>
#include <iostream>
#include <string.h>
 

int cameraId;
//First order cumulative
float prbn;
//Second order cumulative
float meanitr;
//Global mean level
float meanglb;
//Optimum threshold value
int optThresVal;
//Parameters required to work out Otsu's Thresholding algorithm
float param1, param2, param3;

void initParams();

static void captureFrame(std::vector<cv::Mat>& inImages, std::vector<std::string>& imageNames);

void calcHistogram(cv::Mat& frame, cv::Mat& histNorm);

void calcThresholded(cv::Mat& frame, cv::Mat& otsuFrame, cv::Mat& hist);

void detectEdges(cv::Mat& frame, cv::Mat& edgedFrame);

void showImages(cv::Mat& frame, cv::Mat& otsuFrame, cv::Mat& edgedFrame);
