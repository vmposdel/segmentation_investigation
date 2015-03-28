#include "include/depth_experiments.h"

using namespace std;
using namespace cv;

Mat sigma;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
int thresh = 18;
int max_thresh = 40;
RNG rng(12345);
int gaussiansharpenblur = 4;
int maxgaussiansharpenblur = 8;
int debugShow = 1;
std::stringstream imgName;
std::vector<cv::Mat> inImages(203);
std::vector<std::string> imageNames;
cv::Mat image;
double bigContourThresh = 15000;
double hugeContourThresh = 20000;
int lowerContourNumberToTestHuge = 4;
double smallContourThresh = 1500;
float intensityThresh = 100.0;
FILE *fpw;
int img;
int growingPx = 50;
double hasRoiThresh = 70;
double prevPercentThresh = 1.3;
int borderThresh = 20;
int validateThresh = 150;
int lowerWhitesThresh = 20;
int higherWhitesThresh = validateThresh * validateThresh * 0.8;
bool secondPass = false;
std::vector<bool> realContours;
int neighborThresh = 60;
int rayDiffThresh = 0.5 * validateThresh;

static void captureFrame()
{
    tinydir_dir dir;
    tinydir_open(&dir, "/home/v/Documents/Pandora_Vision/dataset_depth");
    cv::Mat tempImg;
    std::stringstream imgName;
    int i = 0;
    while (dir.has_next)
    {
        tinydir_file file;
        tinydir_readfile(&dir, &file);
        if(strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0)
        {
            imgName << "/home/v/Documents/Pandora_Vision/dataset_depth/" << file.name;
            tempImg = cv::imread(imgName.str(), CV_LOAD_IMAGE_COLOR);
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

bool validateContours(cv::Point2f* mc, int ci, std::vector<Point2f>& mcv)
{
    if(!secondPass)
    {
        int sumWhites = 0;
        if(cv::contourArea(contours[ci]) < hugeContourThresh)
        {
            for(int i = mc->y - validateThresh; i < mc->y + validateThresh; i ++)
                for(int j = mc->x - validateThresh; j < mc->x + validateThresh; j ++)
                    if(j >= 0 && j <= sigma.cols && i >= 0 && i <= sigma.rows )
                    {
                        if((int)sigma.at<uchar>(j, i) == 255)
                        {
                            sumWhites ++;
                        }
                    }
            bool horizontalSerie = true;
            int horizontalMaxRay = 0;
            for(int i = mc->y - validateThresh; i < mc->y + validateThresh; i ++)
            {
                int horizontalSum = 0;
                horizontalSerie = true;
                for(int j = mc->x - validateThresh; j < mc->x + validateThresh; j ++)
                {
                    if(j >= 0 && j <= sigma.cols && i >= 0 && i <= sigma.rows )
                    {
                        horizontalSum ++;
                        if((int)sigma.at<uchar>(j, i) != 255)
                        {
                            horizontalSerie = false;
                            break;
                        }
                    }
                    else
                    {
                        horizontalSerie = false;
                    }
                }
                if(horizontalSum > horizontalMaxRay)
                    horizontalMaxRay = horizontalSum;
                if(horizontalSerie)
                    break;
            }
            bool verticalSerie = true;
            int verticalMaxRay = 0;
            for(int i = mc->x - validateThresh; i < mc->x + validateThresh; i ++)
            {
                int verticalSum = 0;
                verticalSerie = true;
                for(int j = mc->y - validateThresh; j < mc->y + validateThresh; j ++)
                {
                    if(j >= 0 && j <= sigma.cols && i >= 0 && i <= sigma.rows )
                    {
                        verticalSum ++;
                        if((int)sigma.at<uchar>(j, i) != 255)
                        {
                            verticalSerie = false;
                            break;
                        }
                    }
                    else
                    {
                        verticalSerie = false;
                    }
                }
                if(verticalSum > verticalMaxRay)
                    verticalMaxRay = verticalSum;
                if(verticalSerie)
                    break;
            }
            if(horizontalSerie || verticalSerie)
            cout << abs(horizontalMaxRay - verticalMaxRay) << "\n";

            if(sumWhites > lowerWhitesThresh && sumWhites < higherWhitesThresh && ((!horizontalSerie && !verticalSerie) || abs(horizontalMaxRay - verticalMaxRay) < rayDiffThresh))
                return true;
            else
                return false;
        }
        else
            return false;
    }
    else
    {
        for(int i = 0; i < contours.size(); i ++)
        {
            if(i != ci)
                if(realContours.at(i) && (abs(mc->x - mcv[i].x) < neighborThresh) && (abs(mc->y - mcv[i].y) < neighborThresh))
                {
                    //cout << realContours.at(i) << "," << abs(mc.x - mcv[i].x) << "," << abs(mc.y - mcv[i].y) << "\n";
                    mc->x = (mc->x + mcv[i].x) / 2;
                    mc->y = (mc->y + mcv[i].y) / 2;
                    realContours.at(i) = false;
                }
        }
        return true;
    }

}

/** @function thresh_callback */
void thresh_callback(int img, void*)
{
    cv::Mat tempSigma;

    //cv::Laplacian(tempSigma, canny_output, CV_8UC1);
    //cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 40));
    ////cv::morphologyEx( canny_output, canny_output, cv::MORPH_OPEN, structuringElement );
    //cv::morphologyEx( canny_output, canny_output, cv::MORPH_CLOSE, structuringElement );
    //cv::imshow("tempSigma1", canny_output);
    //cv::waitKey();
    /// Find contours
    //cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(thresh, thresh));
    //cv::erode(sigma, tempSigma, structuringElement);
    //cv::dilate(sigma, tempSigma, structuringElement);                              
    sigma.copyTo(tempSigma);
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(thresh, thresh));
    //cv::erode(sigma, tempSigma, structuringElement);
    cv::dilate(sigma, tempSigma, structuringElement);
    findContours( tempSigma, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<Moments> mu(contours.size());
    vector<Point2f> mc(contours.size());
    //Mass center
    realContours.clear();
    for(int i = 0; i < contours.size(); i ++)
    {
        mu[i] = moments( contours[i], false );
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
        realContours.push_back(validateContours(&mc[i], i, mc));
    }
    secondPass = true;
    for(int i = 0; i < contours.size(); i ++)
    {
        if(realContours.at(i))
        {
            realContours.at(i) = validateContours(&mc[i], i, mc);
        }
    }
    secondPass = false;
    if(debugShow)
    {
        /// Draw contours
        Mat drawing = Mat::zeros( sigma.size(), CV_8UC3 );
        int j = 0;
        for( int i = 0; i< contours.size(); i++ )
        {
            if(realContours.at(i))
            {
                Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
                //cout << "contour: " << i << ", size: " << contours[i].size() << "\n";
                fprintf(fpw, "%s %d %f %f %f\n", imageNames.at(img).c_str(), j, mc[i].x, mc[i].y, cv::contourArea(contours[i]));
                j ++;                                                              
            }
        }

        //Mass center
        for( int i = 0; i < contours.size(); i++ ){
            if(realContours.at(i))
            {
                //cout << "Mass center: " << (int)mc[i].x << ", " << (int)mc[i].y << "\n";
                std::stringstream label;
                label << cv::contourArea(contours[i]);

                cv::circle(drawing ,cvPoint(mc[i].x, mc[i].y), 5, CV_RGB(0,255,0), -1);
                cv::putText(drawing, label.str(), cvPoint(mc[i].x, mc[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
                label.str("");
            }
        }

        /// Show in a window

        //cout << "Contours: " << contours.size() << "\n";
        //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        //imshow( "Contours", drawing );
        imgName.str("");                                                           
        imgName << "../../../dataset_depth_contours/" << imageNames.at(img);
        cv::imwrite( imgName.str(), drawing );
        imgName.str("");                                                           
    }
}

int main()
{
    fpw = fopen("../../../dataset_depth_contours/results.txt", "w+");
    struct timeval startwtime, endwtime;
    double seq_time;
    gettimeofday (&startwtime, NULL);
    imageNames.clear();
    captureFrame();
    for( img = 0; img < imageNames.size(); img ++ )
    {
        inImages[img].copyTo(image);

        if(!image.data)
        {
            cout << "Cannot open image \n";
            return 0;
        }                                                                          
        cv::cvtColor(image, image, CV_BGR2GRAY);
        image.copyTo(sigma);

        //image.copyTo(sigma);
        //gettimeofday (&endwtime, NULL);
        //seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
        //        + endwtime.tv_sec - startwtime.tv_sec);
        //cout << "Std calculation time: " << seq_time << "\n";

        cv::threshold(sigma, sigma, 1, 255, CV_THRESH_BINARY_INV);
        cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
        cv::morphologyEx( sigma, sigma, cv::MORPH_OPEN, structuringElement );
        structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20));
        cv::morphologyEx( sigma, sigma, cv::MORPH_CLOSE, structuringElement );
        for(int i = 0; i < borderThresh; i ++)
            for(int j = 0; j < sigma.cols; j ++)
                sigma.at<uchar>(i, j) = 0;
        for(int i = sigma.rows - borderThresh; i < sigma.rows; i ++)
            for(int j = 0; j < sigma.cols; j ++)
                sigma.at<uchar>(i, j) = 0;
        for(int i = 0; i < sigma.rows; i ++)
            for(int j = 0; j < borderThresh; j ++)
                sigma.at<uchar>(i, j) = 0;
        for(int i = 0; i < sigma.rows; i ++)
            for(int j = sigma.cols - borderThresh; j < sigma.cols; j ++)
                sigma.at<uchar>(i, j) = 0;
        if(debugShow)
        {
            cv::Mat aggImage;        
            cv::add(image, sigma, aggImage);
            //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
            //imshow( "Contours", drawing );                                       
            std::vector<int> qualityType;
            qualityType.push_back(CV_IMWRITE_JPEG_QUALITY);
            qualityType.push_back(100);
            imgName << "../../../dataset_depth_std_variance/" << imageNames.at(img);
            cv::imwrite( imgName.str(), sigma, qualityType );
            imgName.str("");
            //cv::add(aggImage, sigma, aggImage);
            //char* source_window = "Source";

            //namedWindow( source_window, CV_WINDOW_AUTOSIZE );
            //imshow(source_window, sigma);
            //imshow("Agg_sigm", aggImage);
            createTrackbar( " Dilation Kernel:", "Source", &thresh, max_thresh, thresh_callback );
        }
        thresh_callback( img, 0);
    }

    //imwrite("../../negative43_std.jpg", sigma);
    fclose(fpw);
    waitKey(0);
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);
    cout << "Std plus contour calculation time: " << seq_time << "\n";

    return 0;
}
