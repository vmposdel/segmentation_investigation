#include "include/otsu_thresholding_experimental.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    cv::Mat frame;
    cv::Mat otsuFrame; 
    cv::Mat hist;
    int key = 1;
    initParams();
    while(key != 'q')
    {
        captureFrame(frame);
        const clock_t beginTime = clock();
        if(frame.channels() == 3)
            cv::cvtColor(frame, frame, CV_BGR2GRAY);
        calcHistogram(frame, hist);
        calcThresholded(frame, otsuFrame, hist);
        double execTime = ( clock() - beginTime ) /  static_cast<double>(CLOCKS_PER_SEC );
        printf("Time to execute: %f", execTime);
        showImages(frame, otsuFrame);
        key = cv::waitKey(1);
    }
}

void initParams()
{
     cameraId = 0;
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

void showImages(cv::Mat& frame, cv::Mat& otsuFrame)
{
    cv::namedWindow("OriginalImage", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("OtsuThresholded", CV_WINDOW_AUTOSIZE);
    cv::imshow("OriginalImage", frame);
    cv::imshow("OtsuThresholded", otsuFrame);
}
