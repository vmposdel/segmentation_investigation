#include "include/otsu_thresholding_experimental.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    cv::Mat frame;
    cv::Mat otsuFrame; 
    //cv::Mat hist;
    float hist[256];
    int key = 1;
    initParams();
    while(key != 'q')
    {
        captureFrame(frame);
        if(frame.channels() == 3)
            cv::cvtColor(frame, frame, CV_BGR2GRAY);
        calcHistogram(frame, hist);
        calcThresholded(frame, otsuFrame, hist);
        showImages(frame, otsuFrame);
        key = cv::waitKey(1);
    }
}

void initParams()
{
     cameraId = 1;
     //First order cumulative
     prbn = 0.0;
     //Second order cumulative
     meanitr = 0.0;
     //Global mean level
     meanglb = 0.0;
     //Optimum threshold value
     optThresVal = 0;
     //One of the parameters required to work out Otsu's Thresholding algorithm
     param3 = 0.0;
}

void captureFrame(cv::Mat& frame)
{
    cv::VideoCapture camera(cameraId);
    cv::Mat destFrame;

     // Get the next frame.
    camera.grab();
    camera.retrieve(destFrame);
    frame = destFrame;
    //cv::cvtColor( _dest_img, _hsv_dest_img, CV_BGR2HSV);
    //_hue_dest_ch.create( _hsv_dest_img.size(), _hsv_dest_img.depth() );
    //cv::mixChannels( &_hsv_dest_img, 1, &_hue_dest_ch, 1, ch, 1);
    //cv::calcBackProject( &_hue_dest_ch, 1, 0, _hist, _backproj, &_ranges, 1, true );
    ///// Draw the backproj
    //cv::imshow( "BackProj", _backproj );
    //detectEdges( _backproj );
    //if( cv::waitKey(10)>10 ) break;
}

void calcHistogram(cv::Mat& frame,  float* histNorm)
{
    int hist[256];  
    //position or pixel value of the image
    int pixelPos; 
    int h = frame.rows;
    int w = frame.cols;
    for(int j = 0; j < frame.rows; ++j)
    {
        uchar* histt =  (uchar*) (frame.data + j * frame.step);
        for(int i = 0; i < frame.cols; i++)
        {
            pixelPos = histt[i];
            hist[pixelPos] += 1;    
        }
    }

    for(int i = 0; i < 256; ++i)
    {
        histNorm[i] = hist[i] / (float)(w * h);
        meanglb += ((float)i * histNorm[i]);
    }
    //array to store histogram
    //cv::Mat hist;
    //int channels[] = {0};
    //int  histSize[] = {32};
    //float range[] = {0, 256};
    //const float* histRange[] = {range};    
	//cv::calcHist(&frame, 1, channels, cv::Mat(), hist, 1, histSize, histRange, true, false);
    //histNorm = hist / (frame.rows * frame.cols);
}

void calcThresholded(cv::Mat& frame, cv::Mat& otsuFrame, float* hist)
{
   //First order cumulative
   for(int i = 0; i <= 255; i++)
   {
       prbn += static_cast<float>(hist[i]);
       meanitr += static_cast<float>(i * hist[i]);
       param1 = static_cast<float>((meanglb * prbn) - meanitr);
       param2 = static_cast<float>(param1 * param1) / static_cast<float>(prbn * (1.0f - prbn));
       if(param2 > param3)
       {
           param3 = param2;
           optThresVal = i;
       }
   }
   //Threshold
   cv::threshold(frame, otsuFrame, optThresVal, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

   
}

void showImages(cv::Mat& frame, cv::Mat& otsuFrame)
{
    cv::namedWindow("OriginalImage", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("OtsuThresholded", CV_WINDOW_AUTOSIZE);
    cv::imshow("OriginalImage", frame);
    cv::imshow("OtsuThresholded", otsuFrame);
}
