#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>

using namespace std;
using namespace cv;

Mat sigma;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
int gaussiansharpenblur = 1;
int maxgaussiansharpenblur = 8;
int debugShow = 0;

Mat mat2gray(const Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 1.0, NORM_MINMAX);
    return dst;
}

/** @function thresh_callback */
void thresh_callback(int, void*)
{
    Mat canny_output;
    cv::Mat tempSigma;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    cv::GaussianBlur(sigma, tempSigma, cv::Size(0, 0), gaussiansharpenblur);

    /// Detect edges using canny
    Canny( tempSigma, canny_output, thresh, thresh*2, 3 );
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(40, 40));
    cv::morphologyEx( canny_output, canny_output, cv::MORPH_CLOSE, structuringElement );
    /// Find contours
    findContours( canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    if(debugShow)
    {
        /// Draw contours
        Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
        }

        /// Show in a window
        cout << "Contours: " << contours.size() << "\n";
        namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
        imshow( "Contours", drawing );
    }
}

int main()
{
    struct timeval startwtime, endwtime;
    double seq_time;
    gettimeofday (&startwtime, NULL);

    Mat image = imread("../../negative43.jpg", 0);//cv::IMREAD_ANYDEPTH|cv::IMREAD_ANYCOLOR);
    if(!image.data)
    {
        cout << "Cannot open image \n";
        return 0;
    }
    //// apply median blur
    //cv::Mat tempMedian;
    //int maxKernelLength = 7;
    //for ( int i = 1; i < maxKernelLength; i = i + 2 )
    //    medianBlur (image, image, i);
    //image.copyTo(tempMedian);

    Mat image32f;
    image.convertTo(image32f, CV_32F);

    Mat mu;
    blur(image32f, mu, Size(3, 3));

    Mat mu2;
    blur(image32f.mul(image32f), mu2, Size(3, 3));

    cv::sqrt(mu2 - mu.mul(mu), sigma);

    sigma = mat2gray(sigma);
    //imshow("coke", mat2gray(image32f));
    //imshow("mu", mat2gray(mu));
    //imshow("median", tempMedian);
    double minVal, maxVal;
    minMaxLoc(sigma, &minVal, &maxVal);
    sigma.convertTo(sigma, CV_8U, 255.0/(maxVal - minVal));
    if(debugShow)
    {
        char* source_window = "Source";
        namedWindow( source_window, CV_WINDOW_AUTOSIZE );
        imshow(source_window, sigma);
        createTrackbar( " Canny thresh:", "Source", &thresh, max_thresh, thresh_callback );
        thresh_callback( 0, 0);
    }

    //imwrite("../../negative43_std.jpg", sigma);
    waitKey(0);
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
            + endwtime.tv_sec - startwtime.tv_sec);
    cout << "Std plus contour calculation time: " << seq_time << "\n";

    return 0;
}
