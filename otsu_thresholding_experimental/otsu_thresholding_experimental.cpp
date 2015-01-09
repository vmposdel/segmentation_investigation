#include "include/otsu_thresholding_experimental.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    cv::Mat frame;
    cv::Mat otsuFrame; 
    cv::Mat hist;
    cv::Mat hue_ch;
    int key = 1;
    initParams();
    while(key != 'q')
    {
        captureFrame(frame);
        if(frame.channels() == 3)
           convertImage(frame, hue_ch); 
        calcHistogram(frame, hue_ch, hist);
        hue_ch.copyTo(frame);
        calcThresholded(frame, otsuFrame, hist);
        showImages(frame, otsuFrame);
        key = cv::waitKey(1);
    }
}

void initParams()
{
     cameraId = 0;
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

void convertImage(cv::Mat& frame, cv::Mat& hue_ch)
{
    cv::cvtColor(frame, frame, CV_BGR2HSV);
    hue_ch.create(frame.size(), frame.depth());    
}

void calcHistogram(cv::Mat& frame, cv::Mat& hue_ch, cv::Mat& histNorm)
{
    //array to store histogram
    cv::Mat hist;
    int channels[] = {0, 0};
    int  histSize[] = {128};
    float range[] = {0, 180};
    const float* histRange[] = {range};    
	cv::mixChannels( &frame, 1, &hue_ch, 1, channels, 1);
	cv::calcHist(&hue_ch, 1, 0, cv::Mat(), hist, 1, histSize, histRange, true, false);
    histNorm = hist / (hue_ch.rows * hue_ch.cols);
}

void calcThresholded(cv::Mat& frame, cv::Mat& otsuFrame, cv::Mat& hist)
{
   for(int i = 0; i < 256; i++)
       meanglb += static_cast<float>(i * hist.at<float>(i));
   //First order cumulative
   for(int i = 0; i <= 255; i++)
   {
       prbn += static_cast<float>(hist.at<float>(i));
       meanitr += static_cast<float>(i * hist.at<float>(i));
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
