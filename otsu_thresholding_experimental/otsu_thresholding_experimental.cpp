#include "include/otsu_thresholding_experimental.h"
#include <sys/time.h>

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    cv::Mat frame;
    cv::Mat otsuFrame; 
    cv::Mat edgedFrame;
    cv::Mat hist;
    struct timeval startwtime, endwtime;
    double execTime;  
    int key = 1;
    initParams();
    while(key != 'q')
    {
        captureFrame(frame);
        gettimeofday (&startwtime, NULL);
        //const clock_t beginTime = clock();
        if(frame.channels() == 3)
            cv::cvtColor(frame, frame, CV_BGR2GRAY);
        calcHistogram(frame, hist);
        calcThresholded(frame, otsuFrame, hist);
        gettimeofday (&endwtime, NULL);
        execTime = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
                      + endwtime.tv_sec - startwtime.tv_sec);
        //execTime = (clock() - beginTime) /  static_cast<double>(CLOCKS_PER_SEC );
        printf("Time to execute: %f\n", execTime); 
        detectEdges(otsuFrame, edgedFrame);
        showImages(frame, otsuFrame, edgedFrame);
        key = cv::waitKey(1);
    }
}

void initParams()
{
     cameraId = 1;
}

void captureFrame(cv::Mat& frame)
{
    cv::VideoCapture camera(cameraId);
    cv::Mat destFrame;

     // Get the next frame.
    camera.grab();
    camera.retrieve(destFrame);
    frame = destFrame;
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
}
