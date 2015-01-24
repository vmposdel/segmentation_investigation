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
//    cv::GaussianBlur(frame, frame, cv::Size(5,5), sigma, 0, BORDER_DEFAULT);
    calcHistogram(frame, hist);
    calcThresholded(frame, otsuFrame, hist);
//    detectEdges(otsuFrame, edgedFrame);
    ////printf("Time to execute: %f", execTime);
    //showImages(frame, otsuFrame, edgedFrame);
    ////printf("%s\n", imageNames.at(i));
//    imgName << "../BSDTrainEdged/" << imageNames.at(i);
    //cv::imwrite( imgName.str(), edgedFrame );
//    imgName.str("");
//    imgName << "../BSDTrainThresholded/" << imageNames.at(i);
    //cv::imwrite( imgName.str(), otsuFrame );
//    imgName.str("");
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
      //printf("%s\n", file.name);
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
  int  histSize[] = {512};
  float range[] = {0, 256};
  const float* histRange[] = {range};    
  cv::calcHist(&frame, 1, channels, cv::Mat(), hist, 1, histSize, histRange, true, false);
  cv::normalize(hist, histNorm, 0, 255, NORM_MINMAX, -1, cv::Mat());
}

void calcThresholded(cv::Mat& frame, cv::Mat& otsuFrame, cv::Mat& hist)
{
  //Threshold
  float firstPeak = 0;
  float secondPeak = 0;
  float thirdPeak = 0;
  int peaks[3];
  float firstMean = 0;
  float secondMean = 0;
  int firstMeanI = 0;
  int secondMeanI = 0;
  int maxI = 0;
  int middleI = 0;
  int minI = 0;
  int max = 0;
  int middle = 0;
  int min = 130;
  //float thirdMean = 0;
  for(int i = 0; i < 512; i += 32)
  {
    if(hist.at<float>(i) > firstPeak)
    {
      secondPeak = firstPeak;
      peaks[1] = peaks[0];
      firstPeak = hist.at<float>(i);
      peaks[0] = i;
    }
    else if(hist.at<float>(i) > secondPeak)
    {
      thirdPeak = secondPeak;
      peaks[2] = peaks[1];
      secondPeak = hist.at<float>(i);
      peaks[1] = i;
    }
    else if(hist.at<float>(i) > thirdPeak)
    {
      thirdPeak = hist.at<float>(i);
      peaks[2] = i;
    }
  }
  for(int i = 0; i<3; i++)
  {
    if(peaks[i] > max)
    {
      max = peaks[i];
      maxI = i;
    }
    else if(peaks[i] < min)
    {
      min = peaks[i];
      minI = i;
    }
    else
    {
      middle = peaks[i];
      middleI = i;
    }
  }
  //printf("MinI: %d, MiddleI: %d, MaxI: %d\n", minI, middleI, maxI);
  //firstMeanI = 19;
  //secondMeanI = 26;
  firstMeanI = peaks[minI] + floor((peaks[middleI] - peaks[minI])/2);
  secondMeanI = peaks[middleI] + floor((peaks[maxI] - peaks[middleI])/2);
  //printf("First peak: %d, second peak: %d\n", firstMeanI, secondMeanI);
  int peaksNo = 2;
  if(abs(firstMeanI - secondMeanI) <= 30)
    peaksNo = 1;
  cv::Mat imageT1 = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  imageT1.setTo(peaks[middleI] * 256 / 512);
  cv::Mat imageT2 = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
  imageT2.setTo(peaks[maxI] * 256 / 512);
  ////cv::Mat imageT3;
  //int imgT1I = 0;
  //int rowT1I = 0;
  //int imgT2I = 0;
  //int rowT2I = 0;
  //int imgT3I = 0;
  if(peaksNo == 2)
  {
    for(int y = 0; y < frame.rows; y++)
    {
      for(int x = 0; x < frame.cols; x++)
      {
        // get pixel
        uchar value1 = frame.at<uchar>(y,x);
        int value = (int)value1;
        if(value <= hist.at<float>(secondMeanI))
        {
               // //printf("hist: %f val: %d\n", hist.at<float>(secondMeanI) , value);
          imageT1.at<uchar>(y, x) = value;
          //imgT1I++;
        }
        else // if(value >= hist.at<float>(secondMeanI))
        {
          //     //printf("hist: %f val: %d\n", hist.at<float>(firstMeanI) , value);
          imageT2.at<uchar>(y, x) = value;
          //imgT2I++;
        }
    //    else
    //    {
    //      //    //printf("hist: %f val: %d\n", hist.at<float>(firstMeanI) , value);
    //      imageT1.at<uchar>(x, y) = value1;
    //      imgT1I++;
    //      imageT2.at<uchar>(x, y) = value1;
    //      imgT2I++;
    //    }
      }
    }
    //cv::namedWindow("image1", CV_WINDOW_AUTOSIZE);
    //cv::imshow("image1", imageT1);
    //cv::namedWindow("image2", CV_WINDOW_AUTOSIZE);
    //cv::imshow("image2", imageT2);
    //cv::waitKey(20000);
    //double firstThreshold = cv::threshold(imageT1, imageT1, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    double firstThreshold = 0;
    double secondThreshold = 0;
    //double secondThreshold = cv::threshold(frame, imageT2, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    //printf("First threshold: %f, second threshold: %f\n", firstThreshold, secondThreshold);
    cv::imshow("image1", imageT1);
    cv::imshow("image2", imageT2);
    cv::waitKey(10000);
  }
  else
  {
    double firstThreshold = cv::threshold(frame, otsuFrame, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    //printf("First threshold: %f\n", firstThreshold);
  }
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
