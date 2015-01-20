#include "include/otsu_thresholding_experimental.h"
#include "include/tinydir.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    cv::Mat frame;
    std::vector<cv::Mat> inImages(202);
    std::stringstream imgName;
    std::vector<std::string> imageNames;
    imageNames.clear();
   // char *imageNames[202];
    cv::Mat otsuFrame; 
    cv::Mat edgedFrame; 
    cv::Mat hist;
    initParams();
    captureFrame(inImages, imageNames);
    for( int i = 0; i < imageNames.size(); i++ )
    {
      inImages[i].copyTo(frame);
      if(frame.channels() == 3)
        cv::cvtColor(frame, frame, CV_BGR2GRAY);
      float sigma = 1;
      cv::GaussianBlur(frame, frame, cv::Size(5,5), sigma, 0, BORDER_DEFAULT);
      calcHistogram(frame, hist);
      calcThresholded(frame, otsuFrame, hist);
      detectEdges(otsuFrame, edgedFrame);
      //printf("Time to execute: %f", execTime);
      //showImages(frame, otsuFrame, edgedFrame);
      //printf("%s\n", imageNames.at(i));
      imgName << "../BSDTrainEdged/" << imageNames.at(i);
      cv::imwrite( imgName.str(), edgedFrame );
      imgName.str("");
    }
}

void initParams()
{
     cameraId = 0;
}

static void captureFrame(std::vector<cv::Mat>& inImages, std::vector<std::string>& imageNames)
{
    tinydir_dir dir;
    tinydir_open(&dir, "/home/v/Documents/PANDORA/Vision/Image_Segmentation/BSDS500/BSDS500/data/images/train/");
    std::stringstream imgName;
    int i = 0;
    while (dir.has_next)
    {
      tinydir_file file;
      tinydir_readfile(&dir, &file);
      if(strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0)
      {
        imageNames.push_back(file.name);
        imgName << "/home/v/Documents/PANDORA/Vision/Image_Segmentation/BSDS500/BSDS500/data/images/train/" << file.name;
        inImages[i] = cv::imread(imgName.str());
        imgName.str("");
      }
      tinydir_next(&dir);
      i++;
    }
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
