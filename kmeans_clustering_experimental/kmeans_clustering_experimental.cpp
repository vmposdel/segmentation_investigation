#include "include/kmeans_clustering_experimental.h"
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
  cv::Mat edgedFrame; 
  //cv::Mat hist;
  initParams();
  //  generatePoints();
  captureFrame(inImages, imageNames);
  for( int i = 0; i < imageNames.size(); i++ )
  {
    inImages[i].copyTo(frame);
    if(frame.channels() == 3)
      cv::cvtColor(frame, frame, CV_BGR2GRAY);
    cv::Mat kmeansFrame(frame.cols, frame.rows, CV_8UC3); 
    //float sigma = 1;
    //cv::GaussianBlur(frame, frame, cv::Size(5,5), sigma, 0, BORDER_DEFAULT);
    //calcHistogram(frame, hist);
    calcClustered(frame, kmeansFrame);
    detectEdges(kmeansFrame, edgedFrame);
    //printf("Time to execute: %f", execTime);
    showImages(frame, kmeansFrame, edgedFrame);
    cv::waitKey(10000);
    //printf("%s\n", imageNames.at(i));
    //imgName << "../BSDTrainEdgedKmeans/" << imageNames.at(i);
    //cv::imwrite( imgName.str(), edgedFrame );
    //imgName.str("");
    //imgName << "../BSDTrainClustered/" << imageNames.at(i);
    //cv::imwrite( imgName.str(), kmeansFrame );
    imgName.str("");
  }
}

void initParams()
{
  cameraId = 0;
  maxClusters = 3;
  color[0] = cv::Vec3b(0, 0, 255);
  color[1] = cv::Vec3b(0, 255, 0);
  color[1] = cv::Vec3b(255, 0, 0);
  //colorTab[1] =  cv::Scalar(0, 0, 255);
  //colorTab[2] =  cv::Scalar(0, 255, 0);
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

void calcClustered(cv::Mat& frame, cv::Mat& kmeansFrame)
{
  //Cluster
  int size = frame.rows * frame.cols;
  cv::Mat points(size, 1, CV_32FC1), labels;
  cv::Mat centers(maxClusters, 1, points.type());
  for(int y = 0; y < frame.rows; y ++) {
    for(int x = 0; x < frame.cols; x ++){
      points.at<int>(y * frame.cols + x) = (int)frame.at<uchar>(y, x);
    }
  }
  cv::kmeans(points, maxClusters, labels, 
      cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 1.0),
      10, KMEANS_PP_CENTERS, centers);
  printf("centers:%d, %d\n", centers.at<int>(0), centers.at<int>(1));
  for(int y = 0; y < frame.rows; y ++){
    for(int x = 0; x < frame.cols; x ++){
      //kmeansFrame.at<Vec3b>(Point(y, x)).setTo(colorTab[labels.at<uchar>(y * frame.cols + x)]);
      //printf("label:%d\n", labels.at<int>(y * frame.cols + x));
      //printf("sample:%d\n", points.at<int>(y * frame.cols + x));
      kmeansFrame.at<Vec3b>(Point(y, x)) = color[labels.at<int>(y * frame.cols + x)];
    }
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

void showImages(cv::Mat& frame, cv::Mat& kmeansFrame, cv::Mat& edgedFrame)
{
  cv::namedWindow("OriginalImage", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("KmeansClustered", CV_WINDOW_AUTOSIZE);
  cv::namedWindow("Edges", CV_WINDOW_AUTOSIZE);
  cv::imshow("OriginalImage", frame);
  cv::imshow("KmeansClustered", kmeansFrame);
  cv::imshow("Edges", edgedFrame);
  cv::waitKey(100000);
}
