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
  struct timeval startwtime, endwtime;
  double execTime;
  initParams();
  //  generatesamples();
  gettimeofday (&startwtime, NULL);
  captureFrame(inImages, imageNames);
  for( int i = 0; i < imageNames.size(); i++ )
  {
    inImages[i].copyTo(frame);
    if(!frame.data)
      continue;
    //if(frame.channels() == 3)
    //  cv::cvtColor(frame, frame, CV_BGR2GRAY);
    cv::Mat kmeansFrame(frame.rows, frame.cols, CV_8UC3); 
    //float sigma = 1;
    //cv::GaussianBlur(frame, frame, cv::Size(5,5), sigma, 0, BORDER_DEFAULT);
    //calcHistogram(frame, hist);
    calcClustered(frame, kmeansFrame);
    detectEdges(kmeansFrame, edgedFrame);
    //printf("Time to execute: %f", execTime);
    //showImages(frame, kmeansFrame, edgedFrame);
    //cv::waitKey(10000);
    //imgName << "../../../dataset_kmeansC" << maxClusters << "/" << imageNames.at(i);
    //cv::imwrite( imgName.str(), edgedFrame );
    //imgName << "../BSDTrainClustered/" << imageNames.at(i);
    //cv::imwrite( imgName.str(), kmeansFrame );
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
  maxClusters = 5;
  //colors = new int [maxClusters]; 
  colors = new cv::Vec3b [maxClusters]; 
  for(int i = 0; i < maxClusters; i ++)
  {
    if((i + 1) % 2 == 0)
    {
      colors[i][0] = 0;
      colors[i][1] = 255 / ((i + 1) / 2);
      colors[i][2] = 0;
    }
    else if((i + 1) % 3 == 0)
    {
      colors[i][0] = 0;
      colors[i][1] = 0;
      colors[i][2] = 255 / ((i + 1) / 3);
    }
    else
    {
      colors[i][0] = 255 / (i + 1);
      colors[i][1] = 0;
      colors[i][2] = 0;
    }
    //colors[i] = (255 / (i + 1), 255/ (i + 2), 255/ (i + 3));
  }
  //for(int i = 0; i < maxClusters; i ++)
  //{
  //  colors[i] = 255/(i+1);
  //}
}


static void captureFrame(std::vector<cv::Mat>& inImages, std::vector<std::string>& imageNames)
{
  tinydir_dir dir;
  tinydir_open(&dir, "/home/v/Documents/Pandora_Vision/opencv_traincascade/new_svm_data/data/Test_Negative_Images");
  std::stringstream imgName;
  int i = 0;
  while (dir.has_next)
  {
    tinydir_file file;
    tinydir_readfile(&dir, &file);
    if(strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0)
    {
      imageNames.push_back(file.name);
      imgName << "/home/v/Documents/Pandora_Vision/opencv_traincascade/new_svm_data/data/Test_Negative_Images/" << file.name;
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
  cv::Mat samples(size, 5, CV_32F);
  cv::Mat labels;
  cv::Mat centers(maxClusters, 1, samples.type());
  std::vector<cv::Mat> components;    
  cv::split(frame, components);

  for(int y = 0; y < frame.rows * frame.cols; y ++) 
  {
    samples.at<float>(y,0) = (y / frame.cols) / frame.rows;
    samples.at<float>(y,1) = (y % frame.cols) / frame.cols;
    samples.at<float>(y,2) = components[0].data[y] / 255.0;
    samples.at<float>(y,3) = components[1].data[y] / 255.0;
    samples.at<float>(y,4) = components[2].data[y] / 255.0;
  }

  //for(int y = 0; y < frame.rows; y ++) {
  //  for(int x = 0; x < frame.cols; x ++){
  //    samples.at<int>(y * frame.cols + x) = (int)frame.at<uchar>(y, x);
  //  }
  //}
  cv::kmeans(samples, maxClusters, labels, 
      cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
      3, KMEANS_PP_CENTERS, centers);
  for(int y = 0; y < frame.rows * frame.cols; y ++) 
  {
    //printf("point %d", labels.at<int>(y / frame.cols, y % frame.cols));
    kmeansFrame.at<Vec3b>(Point(y % frame.cols, y / frame.cols)) = colors[labels.at<int>(0, y)];
  }
  //for(int y = 0; y < frame.rows * frame.cols; y ++) 
  //{
  //  kmeansFrame.at<float>(y / frame.cols, y % frame.cols) = (float)(colors[labels.at<int>(0, y)]);
  //}
  //kmeansFrame.convertTo(kmeansFrame, CV_8UC3);

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
