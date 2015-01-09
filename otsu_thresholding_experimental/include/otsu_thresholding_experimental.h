#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <stdio.h>
#include <iostream>
 

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

void captureFrame(cv::Mat& frame);

void convertImage(cv::Mat& frame, cv::Mat& hue_ch);

void calcHistogram(cv::Mat& frame, cv::Mat& hue_ch, cv::Mat& histNorm);

void calcThresholded(cv::Mat& frame, cv::Mat& otsuFrame, cv::Mat& hist);

void showImages(cv::Mat& frame, cv::Mat& otsuFrame);