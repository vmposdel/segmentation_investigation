#include "include/otsu_thresholding_experimental.h"
#include "include/tinydir.h"
#include <sys/time.h>

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
  struct timeval startwtime, endwtime;
  double execTime;
  initParams();
  gettimeofday (&startwtime, NULL);
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
    //detectEdges(otsuFrame, edgedFrame);
    //printf("Time to execute: %f", execTime);
    //showImages(frame, otsuFrame, edgedFrame);
    //printf("%s\n", imageNames.at(i));
    imgName << "../../../dataset_otsu/" << imageNames.at(i);
    cv::imwrite( imgName.str(), otsuFrame );
    imgName.str("");
    //imgName << "../BSDTrainThresholded/" << imageNames.at(i);
    //cv::imwrite( imgName.str(), otsuFrame );
    //imgName.str("");
  }
  gettimeofday (&endwtime, NULL);
  execTime = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
      + endwtime.tv_sec - startwtime.tv_sec);
  printf("Time to execute: %f\n", execTime);
}

void initParams()
{
  cameraId = 0;
}

static void captureFrame(std::vector<cv::Mat>& inImages, std::vector<std::string>& imageNames)
{
  tinydir_dir dir;
  tinydir_open(&dir, "/home/v/Documents/Pandora_Vision/opencv_traincascade/new_svm_data/data/Test_Negative_Images");
  cv::Mat tempImg;
  std::stringstream imgName;
  int i = 0;
  while (dir.has_next)
  {
    tinydir_file file;
    tinydir_readfile(&dir, &file);
    if(strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0)
    {
      imgName << "/home/v/Documents/Pandora_Vision/opencv_traincascade/new_svm_data/data/Test_Negative_Images/" << file.name;
      tempImg = cv::imread(imgName.str());
      if(!tempImg.data)
        continue;
      tempImg.copyTo(inImages[i]);
      imageNames.push_back(file.name);
      imgName.str("");
      i++;
    }
    tinydir_next(&dir);
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
