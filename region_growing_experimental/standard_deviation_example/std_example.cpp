#include <iostream>
#include<opencv2/opencv.hpp>
#include <sys/time.h>
#include "include/tinydir.h"

using namespace std;
using namespace cv;

Mat sigma;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
int thresh = 19;
int max_thresh = 40;
RNG rng(12345);
int gaussiansharpenblur = 4;
int maxgaussiansharpenblur = 8;
int debugShow = 1;
std::stringstream imgName;
std::vector<cv::Mat> inImages(202);
std::vector<std::string> imageNames;

static void captureFrame()
{
    tinydir_dir dir;
    tinydir_open(&dir, "/home/v/Documents/Pandora_Vision/opencv_traincascade/new_svm_data/data/Test_Negative_Images");
    cv::Mat tempImg;
    std::stringstream imgName;
    int i = 0;
    while (dir.has_next)
    {
        tinydir_file file;
        tinydir_readfile(&dir, &file);
        if(strcmp(file.name, ".") != 0 && strcmp(file.name, "..") != 0)
        {
            imgName << "/home/v/Documents/Pandora_Vision/opencv_traincascade/new_svm_data/data/Test_Negative_Images/" << file.name;
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
    findContours( tempSigma, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<Moments> mu(contours.size() );
    vector<Point2f> mc( contours.size() );
    //Mass center
    for( int i = 0; i < contours.size(); i++ ){
        mu[i] = moments( contours[i], false );
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); 
    }

    if(debugShow)
    {
        /// Draw contours
        Mat drawing = Mat::zeros( sigma.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
            //cout << "contour: " << i << ", size: " << contours[i].size() << "\n";
        }

        //Mass center
        for( int i = 0; i < contours.size(); i++ ){ 
            //cout << "Mass center: " << (int)mc[i].x << ", " << (int)mc[i].y << "\n";
            std::stringstream label;
            label << cv::contourArea(contours[i]);
            
            cv::circle(drawing ,cvPoint(mc[i].x, mc[i].y), 5, CV_RGB(0,255,0), -1);
            cv::putText(drawing, label.str(), cvPoint(mc[i].x, mc[i].y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
            label.str("");
        }

        /// Show in a window

        //cout << "Contours: " << contours.size() << "\n";
        //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        //imshow( "Contours", drawing );
        imgName << "../../../../dataset_std_variance_contours/" << imageNames.at(img);
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
    struct timeval startwtime, endwtime;
    double seq_time;
    gettimeofday (&startwtime, NULL);
    imageNames.clear();
    captureFrame();
    for( int img = 0; img < imageNames.size(); img ++ )
    {
        cv::Mat image;
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
        cv::threshold(sigma, sigma, 64, 255, CV_THRESH_BINARY);
        cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
        cv::morphologyEx( sigma, sigma, cv::MORPH_OPEN, structuringElement );
        structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(20, 20));
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
            cv::imwrite( imgName.str(), sigmaPure, qualityType );
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
    waitKey(0);
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);
    cout << "Std plus contour calculation time: " << seq_time << "\n";

    return 0;
}
