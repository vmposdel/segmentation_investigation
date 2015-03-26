#include "include/depth_experiments.h"

using namespace std;
FILE *fpr;
int totalImages = 61;
std::stringstream imageNames;
std::string imgNamePrefix("/home/v/Documents/Pandora_Vision/dataset_depth/depthFrame");

int main()
{
    fpr = fopen("/home/v/Documents/Pandora_Vision/dataset_std_variance_contours_BOD/rgb_contours_aggregation.txt", "r");
    for(int i = 216; i <= 216 + totalImages; i ++)
    {
        imageNames.str("");
        struct timeval startwtime, endwtime;
        double seq_time;
        gettimeofday (&startwtime, NULL);
        imageNames << imgNamePrefix << i << ".png";
        cv::Mat image = cv::imread(imageNames.str(), CV_LOAD_IMAGE_COLOR);//cv::IMREAD_ANYDEPTH|cv::IMREAD_ANYCOLOR);
        int contoursNo;
        int imageId;
        fscanf( fpr, "%d", &imageId);
        fscanf( fpr, "%d", &contoursNo);
        std::vector<int> contourSpecs;
        for(int j = 0; j < 3 * contoursNo; j ++)
        {
            double tempSpec;
            fscanf( fpr, "%f", &tempSpec);
            contourSpecs.push_back((int)tempSpec);
        }
        for(int j = 3 * contoursNo; j < 32; j ++)
        {
            fscanf( fpr, "%d", &imageId);
        }

        if(!image.data)
        {
            cout << "Cannot open image \n";
            return 0;
        }
        for(int ci = 0; ci < contoursNo; ci ++)
        {
            int sizeEst = (int)(sqrt(contourSpecs.at(2 + ci * 3)) / 2);
            if(contourSpecs.at(1 + ci * 3) + sizeEst > 480)
                contourSpecs.at(1 + ci * 3) = 480 - sizeEst;
            if(contourSpecs.at(1 + ci * 3) - sizeEst < 0)
                contourSpecs.at(1 + ci * 3) = 0 + sizeEst;
            if(contourSpecs.at(0 + ci * 3) + sizeEst > 640)
                contourSpecs.at(0 + ci * 3) = 640 - sizeEst;
            if(contourSpecs.at(0 + ci * 3) - sizeEst < 0)
                contourSpecs.at(0 + ci * 3) = 0 + sizeEst;
            cout << "size; " << sizeEst << "\n";
            cv::Mat ROI = image(cv::Rect(contourSpecs.at(0 + ci * 3) - sizeEst, contourSpecs.at(1 + ci * 3) - sizeEst, 2 * sizeEst, 2 * sizeEst));
            //cv::imshow("ROI", ROI);
            //cv::waitKey();
        }
        contourSpecs.clear();
        imageNames.str("");
    }
}


