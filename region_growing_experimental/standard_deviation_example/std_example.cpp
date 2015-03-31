#include <iostream>
#include<opencv2/opencv.hpp>
#include <sys/time.h>
#include "math.h"
#include "include/tinydir.h"
#include "utilities/include/haralickfeature_extractor.h"
#include "stdio.h"

using namespace std;
using namespace cv;

Mat sigma;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
int thresh = 8;
int max_thresh = 40;
RNG rng(12345);
int gaussiansharpenblur = 4;
int maxgaussiansharpenblur = 8;
int debugShow = 1;
std::stringstream imgName;
std::vector<cv::Mat> inImages(203);
std::vector<std::string> imageNames;
cv::Mat image;
double bigContourThresh = 4000;
double hugeContourThresh = 10000;
int lowerContourNumberToTestHuge = 4;
double smallContourThresh = 1500;
float intensityThresh = 150.0;
FILE *fpw;
int img;
int dimSize = 20;

static void captureFrame()
{
    tinydir_dir dir;
    tinydir_open(&dir, "/home/v/Documents/Pandora_Vision/dataset_rgb");
    cv::Mat tempImg;
    std::stringstream imgName;
    int i = 0;
    while (dir.has_next)
    {
        tinydir_file file;
        tinydir_readfile(&dir, &file);
        if(strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0)
        {
            imgName << "/home/v/Documents/Pandora_Vision/dataset_rgb/" << file.name;
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

Mat mat2gray(const Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 1.0, NORM_MINMAX);
    return dst;
}

bool validateContours(cv::Point2f& mc, int ci)
{
    int indX = (int)mc.x;
    int indY = (int)mc.y;
    //int dimSize = (int)sqrt(contourArea(contours[ci])) / 2;
    if(contours.size() > lowerContourNumberToTestHuge && cv::contourArea(contours[ci]) > hugeContourThresh)
        return false;
    else if(cv::contourArea(contours[ci]) > bigContourThresh || cv::contourArea(contours[ci]) < smallContourThresh)
    {
        cv::Mat canvas = cv::Mat::zeros(image.size() ,CV_8UC1);
        drawContours( canvas, contours, ci, cv::Scalar(255, 255, 255), CV_FILLED);
        cv::Mat contourROI;
        image.copyTo(contourROI, canvas);
        int xVertice, yVertice, roiWidthX, roiWidthY;
        if(indX - dimSize >= 0 && indX - dimSize + 2 * dimSize < image.cols)
        {
            xVertice = indX - dimSize;
            roiWidthX = 2 * dimSize;
        }
        else if(indX - dimSize < 0)
        {
            xVertice = 0;
            roiWidthX = 2 * dimSize;
        }
        else
        {
            xVertice = indX - dimSize;
            roiWidthX = image.cols - xVertice;
        }
        if(indY - dimSize >= 0 && indY - dimSize + 2 * dimSize < image.rows)
        {
            yVertice = indY - dimSize;
            roiWidthY = 2 * dimSize;
        }
        else if(indY - dimSize < 0)
        {
            yVertice = 0;
            roiWidthY = 2 * dimSize;
        }
        else
        {
            yVertice = indY - dimSize;
            roiWidthY = image.rows - yVertice;
        }
        cv::Mat ROI = image(cv::Rect(xVertice, yVertice, roiWidthX, roiWidthY));
        cv::Scalar mean = cv::mean(ROI);
        //cv::cvtColor(ROI, ROI, CV_BGR2GRAY);
        //cout << mean[0] << "\n";
        //cv::imshow("curr contour", ROI);
        //cv::waitKey(0);
        HaralickFeaturesExtractor haralickFeaturesDetector_;
        haralickFeaturesDetector_.findHaralickFeatures(ROI);
        std::vector<double> haralickFeatures = haralickFeaturesDetector_.getFeatures();
        if(haralickFeatures[0] > 3 || haralickFeatures[1] > 3000)
            return false;
        //if(mean[0] > intensityThresh)
        //    return false;
    }
    return true;
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
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(thresh, thresh));
    //cv::erode(sigma, tempSigma, structuringElement);
    cv::dilate(sigma, tempSigma, structuringElement);
    //sigma.copyTo(tempSigma);
    findContours( tempSigma, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<Moments> mu(contours.size() );
    vector<Point2f> mc( contours.size() );
    std::vector<bool> realContours;
        std::vector<vector<Point> > contours_poly( contours.size() );
        std::vector<Rect> boundRect( contours.size() );
        std::vector<Point2f>center( contours.size() );
        std::vector<float>radius( contours.size() );
    //Mass center
    for( int i = 0; i < contours.size(); i++ ){
        mu[i] = moments( contours[i], false );
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); 
        realContours.push_back(validateContours(mc[i], i));
                cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
                boundRect[i] = cv::boundingRect( cv::Mat(contours_poly[i]) );
                cv::minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
    }

    if(debugShow)
    {
        /// Draw contours
        Mat drawing = Mat::zeros( sigma.size(), CV_8UC3 );
        int j = 0;
        for( int i = 0; i< contours.size(); i++ )
        {
            if(realContours.at(i))
            {
                cv::Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                drawContours( drawing, contours_poly, i, color, 2, 8, hierarchy, 0, Point() );
                //cout << "contour: " << i << ", size: " << contours[i].size() << "\n";
                cv::rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
                //cv::circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
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
        imgName << "../../../../dataset_std_variance_contours_BOD_new_haralick/" << imageNames.at(img);
        cv::imwrite( imgName.str(), drawing );
        imgName.str("");
    }
}

cv::Mat& subtractBackground(cv::Mat& image)
{
    cv::Mat foreground;
    return foreground;
}

int main()
{
    fpw = fopen("../../../../dataset_std_variance_contours_BOD_new_haralick/results.txt", "w+");
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
        //blur(image, image, Size(gaussiansharpenblur, gaussiansharpenblur));
        cv::GaussianBlur(image, image, cv::Size(0, 0), gaussiansharpenblur);
        //// apply median blur
        //cv::Mat tempMedian;
        //int maxKernelLength = 7;
        //for ( int i = 1; i < maxKernelLength; i = i + 2 )
        //    medianBlur (image, image, i);
        //image.copyTo(tempMedian);
        //image = subtractBackground(image);

        Mat image32f;
        image.convertTo(image32f, CV_32F);

        Mat mu;
        blur(image32f, mu, Size(3, 3));

        Mat mu2;
        blur(image32f.mul(image32f), mu2, Size(3, 3));

        cv::sqrt(mu2 - mu.mul(mu), sigma);
        //cv::Mat mean;
        //cv::meanStdDev(image, mean, sigma);

        sigma = mat2gray(sigma);
        cv::cvtColor(sigma, sigma, CV_BGR2GRAY);
        cv::cvtColor(image, image, CV_BGR2GRAY);

        //image.copyTo(sigma);
        //gettimeofday (&endwtime, NULL);
        //seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
        //        + endwtime.tv_sec - startwtime.tv_sec);
        //cout << "Std calculation time: " << seq_time << "\n";

        double minVal, maxVal;
        minMaxLoc(sigma, &minVal, &maxVal);
        sigma.convertTo(sigma, CV_8UC1, 255.0/(maxVal - minVal));
        minMaxLoc(sigma, &minVal, &maxVal);
        cv::Mat sigmaPure;
        sigma.copyTo(sigmaPure);
        //cout << minVal << ", " << maxVal << "\n";
        //image.copyTo(sigma);
        cv::threshold(sigma, sigma, 80, 255, CV_THRESH_BINARY);
        cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));
        cv::morphologyEx( sigma, sigma, cv::MORPH_CLOSE, structuringElement );
        structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(8, 8));
        cv::morphologyEx( sigma, sigma, cv::MORPH_OPEN, structuringElement );
        if(debugShow)
        {
            cv::Mat aggImage;        
            cv::add(image, sigma, aggImage);
            //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
            //imshow( "Contours", drawing );
            std::vector<int> qualityType;
            qualityType.push_back(CV_IMWRITE_JPEG_QUALITY);
            qualityType.push_back(100);
            imgName << "../../../../dataset_std_variance/" << imageNames.at(img);
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
