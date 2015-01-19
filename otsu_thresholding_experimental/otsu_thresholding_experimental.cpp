#include "include/otsu_thresholding_experimental.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    cv::Mat frame;
    cv::Mat otsuFrame; 
    cv::Mat edgedFrame; 
    cv::Mat hist;
    int key = 1;
    initParams();
    captureFrame(frame);
    clock_t beginTime = clock();
    if(frame.channels() == 3)
        cv::cvtColor(frame, frame, CV_BGR2GRAY);
    float sigma = 1;
    cv::GaussianBlur(frame, frame, cv::Size(5,5), sigma, 0, BORDER_DEFAULT);
    calcHistogram(frame, hist);
    calcThresholded(frame, otsuFrame, hist);
    double execTime = static_cast<double>(clock() - beginTime) /  static_cast<double>(CLOCKS_PER_SEC );
    detectEdges(otsuFrame, edgedFrame);
    printf("Time to execute: %f", execTime);
    showImages(frame, otsuFrame, edgedFrame);
    key = cv::waitKey(1);
}

void initParams()
{
     cameraId = 0;
}

void captureFrame(cv::Mat& frame)
{
    frame = cv::imread("/home/v/Documents/PANDORA/Vision/Image Segmentation/BSDS300/images/train/189011.jpg");
}

void calcHistogram(cv::Mat& frame,  cv::Mat& histNorm)
{
    //array to store histogram
    cv::Mat hist;
    int channels[] = {0};
    int  histSize[] = {32};
    float range[] = {0, 256};
    const float* histRange[] = {range};    
    cv::calcHist(&frame, 1, channels, cv::Mat(), hist, 1, histSize, histRange, true, false);
    cv::normalize(hist, histNorm, 0, 255, NORM_MINMAX, -1, cv::Mat());
}

void calcThresholded(cv::Mat& frame, cv::Mat& otsuFrame, cv::Mat& hist)
{
   //Threshold
   cv::threshold(frame, otsuFrame, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
}

void detectEdges(cv::Mat& frame, cv::Mat& edgedFrame)
{
  cv::Mat dstImg;
  cv::Mat dst;
  int thresholdLow = 10;
  int ratio = 3;
  float sigma = 1;
  cv::GaussianBlur(frame, dstImg, cv::Size(5,5), sigma, 0, BORDER_DEFAULT);
  cv::Canny(dstImg, dstImg, thresholdLow, ratio * thresholdLow);
  dstImg.copyTo(edgedFrame);
}

void showImages(cv::Mat& frame, cv::Mat& otsuFrame, cv::Mat& edgedFrame)
{
    cv::namedWindow("OriginalImage", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("OtsuThresholded", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Edges", CV_WINDOW_AUTOSIZE);
    cv::imshow("OriginalImage", frame);
    cv::imshow("OtsuThresholded", otsuFrame);
    cv::imshow("Edges", edgedFrame);
    cv::waitKey(100000);
}
