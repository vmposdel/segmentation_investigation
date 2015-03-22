#include "include/depth_experiments.h"

using namespace std;

int main()
{
    struct timeval startwtime, endwtime;
    double seq_time;
    gettimeofday (&startwtime, NULL);
    std::string imgName = "../depthFrame0239";
    std::stringstream imgNames;
    imgNames.str("");
    imgNames << imgName << ".png";
    cv::Mat image = cv::imread(imgNames.str(), CV_LOAD_IMAGE_COLOR);//cv::IMREAD_ANYDEPTH|cv::IMREAD_ANYCOLOR);
    int contoursNo = 7;
    int contoursCenterY[7] = {99, 287, 443, 21, 161, 411, 237};
    int contoursCenterX[7] = {53, 125, 39, 248, 320, 397, 626};
    int sizeEst[7] = {40, 80, 21, 17, 52, 70, 17};

    if(!image.data)
    {
        cout << "Cannot open image \n";
        return 0;
    }
    for(int i = 0; i < contoursNo; i ++)
    {
        if(contoursCenterY[i] + sizeEst[i] > 480)
            contoursCenterY[i] = 480 - sizeEst[i];
        if(contoursCenterY[i] - sizeEst[i] < 0)
            contoursCenterY[i] = 0 + sizeEst[i];
        if(contoursCenterX[i] + sizeEst[i] > 640)
            contoursCenterX[i] = 640 - sizeEst[i];
        if(contoursCenterX[i] - sizeEst[i] < 0)
            contoursCenterX[i] = 0 + sizeEst[i];
        cout << "size; " << sizeEst[i] << "\n";
        cv::Mat ROI = image(cv::Rect(contoursCenterX[i] - sizeEst[i], contoursCenterY[i] - sizeEst[i], 2 * sizeEst[i], 2 * sizeEst[i]));
        cv::imshow("ROI", ROI);
        cv::waitKey();
    }
}


