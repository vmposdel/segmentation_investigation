#include "include/depth_experiments.h"

using namespace std;
FILE *fpr;
int totalImages = 61;
std::stringstream imageNames;
std::string imgNamePrefix("/home/v/Documents/Pandora_Vision/dataset_depth/depthFrame0");

int main()
{
    fpr = fopen("/home/v/Documents/Pandora_Vision/dataset_std_variance_contours_BOD_new/rgb_contours_aggregation.txt", "r");
    for(int i = 216; i <= 216 + totalImages; i ++)
    {
        imageNames.str("");
        std::stringstream outImageNames;
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
            fscanf( fpr, "%lf", &tempSpec);
            contourSpecs.push_back((int)tempSpec);
            //cout << "contoArea" << contourSpecs.at(2 + ci * 3) << "\n";
        }
        for(int j = 3 * contoursNo + 2; j < 32; j ++)
        {
            fscanf( fpr, "%d", &imageId);
        }

        if(!image.data)
        {
            cout << "Cannot open image \n";
            continue;
        }
        for(int ci = 0; ci < contoursNo; ci ++)
        {
            cout << "Contour: " << ci << "of image " << i << "\n";
            int sizeEst = (int)(sqrt(contourSpecs.at(2 + ci * 3)) / 1);
            if(contourSpecs.at(1 + ci * 3) + sizeEst > 480)
                contourSpecs.at(1 + ci * 3) = 480 - sizeEst;
            if(contourSpecs.at(1 + ci * 3) - sizeEst < 0)
                contourSpecs.at(1 + ci * 3) = 0 + sizeEst;
            if(contourSpecs.at(0 + ci * 3) + sizeEst > 640)
                contourSpecs.at(0 + ci * 3) = 640 - sizeEst;
            if(contourSpecs.at(0 + ci * 3) - sizeEst < 0)
                contourSpecs.at(0 + ci * 3) = 0 + sizeEst;
            cout << "size; " << sizeEst << ", " << i << "\n";
            cv::Mat ROI = image(cv::Rect(contourSpecs.at(0 + ci * 3) - sizeEst, contourSpecs.at(1 + ci * 3) - sizeEst, 2 * sizeEst, 2 * sizeEst));
            outImageNames << "/home/v/Documents/Pandora_Vision/depth_ROI/frame" << i << "_" << ci << ".png";
            cv::imwrite(outImageNames.str(), ROI);
            outImageNames.str("");
            cv::Canny(ROI, ROI, 10, 3 * 10);
            outImageNames << "/home/v/Documents/Pandora_Vision/depth_ROI_edges/frame" << i << "_" << ci << ".png";
            cv::imwrite(outImageNames.str(), ROI);
            outImageNames.str("");

            //cv::imshow("ROI", ROI);
            //cv::waitKey();
        }
        contourSpecs.clear();
        imageNames.str("");
    }
}


