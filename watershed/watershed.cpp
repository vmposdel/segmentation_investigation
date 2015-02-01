#include "opencv2/opencv.hpp"
#include <string>
#include <stdio.h>
#include "include/tinydir.h"
#include <sys/time.h>

using namespace cv;
using namespace std;

class WatershedSegmenter{
  private:
    cv::Mat markers;
  public:
    void setMarkers(cv::Mat& markerImage)
    {
      markerImage.convertTo(markers, CV_32S);
    }

    cv::Mat process(cv::Mat &image)
    {
      cv::watershed(image, markers);
      markers.convertTo(markers,CV_8U);
      return markers;
    }
};

static void captureFrame(std::vector<cv::Mat>& inImages, std::vector<std::string>& imageNames);

/*Disable imshows,saves the segmented images.
 * Need to save inputs as 0-10 in the end
 */
int main(int argc, char* argv[])
{
  cv::Mat image;
  std::vector<cv::Mat> inImages(202);
  std::stringstream imgName;
  std::vector<std::string> imageNames;
  struct timeval startwtime, endwtime;
  double execTime;
  imageNames.clear();
  int i = 1;
  gettimeofday (&startwtime, NULL);
  captureFrame(inImages, imageNames);
  for( int i = 0; i < imageNames.size(); i++ ) 
  {
    inImages[i].copyTo(image);

    if(!image.data)
    {
      cout<<"error while reading image\n";
      continue;
    }
    cv::Mat binary;
    cv::cvtColor(image, binary, CV_BGR2GRAY);
    cv::threshold(binary, binary, 100, 255, THRESH_BINARY);

    //imshow("originalimage", image);
    //imshow("originalbinary", binary);

    // Eliminate noise and smaller objects
    cv::Mat fg;
    cv::erode(binary,fg,cv::Mat(),cv::Point(-1,-1),2);
    //imshow("fg", fg);

    // Identify image pixels without objects
    cv::Mat bg;
    cv::dilate(binary,bg,cv::Mat(),cv::Point(-1,-1),3);
    cv::threshold(bg,bg,1, 128,cv::THRESH_BINARY_INV);
    //imshow("bg", bg);

    // Create markers image
    cv::Mat markers(binary.size(),CV_8U,cv::Scalar(0));
    markers= fg+bg;
    //imshow("markers", markers);

    // Create watershed segmentation object
    WatershedSegmenter segmenter;
    segmenter.setMarkers(markers);

    cv::Mat result = segmenter.process(image);
    result.convertTo(result,CV_8U);
    //imshow("final_result", result);
    //cv::waitKey(20000);
    //sprintf(name,"waterResult%d.jpeg",i);
    //imwrite(name,result);
    imgName << "../../../dataset_WatershedMarkers/" << imageNames.at(i);
    cv::imwrite( imgName.str(), markers );
    imgName.str("");
  }
  gettimeofday (&endwtime, NULL);
  execTime = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
      + endwtime.tv_sec - startwtime.tv_sec);
  printf("Time to execute: %f\n", execTime);
  cv::waitKey(0);

  return 0;
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
