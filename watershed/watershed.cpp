#include "opencv2/opencv.hpp"
#include <string>
#include <stdio.h>

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

/*Disable imshows,saves the segmented images.
 * Need to save inputs as 0-10 in the end
 */
int main(int argc, char* argv[])
{
  char name[10];
  int i=0;
  while(1)
  {
    sprintf(name, "img%d.jpeg",i);
    cv::Mat image = cv::imread(name,1);
    
    if(!image.data)
    {
      cout<<"error while reading image\n";
      break;
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
    sprintf(name,"waterResult%d.jpeg",i);
    imwrite(name,result);
    i++;
  }
  cv::waitKey(0);

  return 0;
}
