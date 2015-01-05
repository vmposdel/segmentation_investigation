#include "include/histogram_edge_blobs.h"

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    std::vector<cv::Mat> _in_images(7);
    std::stringstream _img_name;
    for( int i=0; i<_in_images.size(); i++ ){
        _img_name << "../walls/" << i << ".png";
        _in_images[i] = cv::imread(_img_name.str());
        _img_name.str("");
    	//cv::Mat _in_img;
	//std::string _input_img("../walls/1.png");
        //_in_image2 = cv::imread(_input_image2, 1);
        
    }
    //calcHistogram(_in_image);
    calcBackProjection( _in_images );
//    detectEdges(_in_image);
//    detectBlobs(_in_image);
      
}

void calcHistogram(const cv::Mat& _in_image)
{
    cv::vector<cv::Mat> _bgr_parts;
    cv::split(_in_image, _bgr_parts);
    //set number of bins
    int _hist_sizes = 128;
    //set ranges for b,g,r
    float _range[] = {0, 256};
    const float* _hist_range = {_range};    
    cv::Mat _b_hist, _g_hist, _r_hist;
    cv::calcHist(&_bgr_parts[0], 1, 0, cv::Mat(), _b_hist, 1, &_hist_sizes, &_hist_range, true, false);
    cv::calcHist(&_bgr_parts[1], 1, 0, cv::Mat(), _g_hist, 1, &_hist_sizes, &_hist_range, true, false);
    cv::calcHist(&_bgr_parts[2], 1, 0, cv::Mat(), _r_hist, 1, &_hist_sizes, &_hist_range, true, false);
    //draw the histograms
    int _hist_width = 512, _hist_height = 400;
    int _bin_width = cvRound( (double) _hist_width/_hist_sizes );
    cv::Mat _hist_img ( _hist_height, _hist_width, CV_8UC3, cv::Scalar( 0, 0 ,0) );
    //Normalize to fall into range
    normalize(_b_hist, _b_hist, 0, _hist_img.rows, NORM_MINMAX, -1, cv::Mat() );
    normalize(_g_hist, _g_hist, 0, _hist_img.rows, NORM_MINMAX, -1, cv::Mat() );
    normalize(_r_hist, _r_hist, 0, _hist_img.rows, NORM_MINMAX, -1, cv::Mat() );
    for( int i = 1; i < _hist_sizes; i++ )
    {
        line( _hist_img, Point( _bin_width*(i-1), _hist_height - cvRound(_b_hist.at<float>(i-1)) ) , Point( _bin_width*(i), _hist_height - cvRound(_b_hist.at<float>(i)) ), Scalar( 255, 0, 0), 2, 8, 0  );
        line( _hist_img, Point( _bin_width*(i-1), _hist_height - cvRound(_g_hist.at<float>(i-1)) ) , Point( _bin_width*(i), _hist_height - cvRound(_g_hist.at<float>(i)) ), Scalar( 0, 255, 0), 2, 8, 0  );
        line( _hist_img, Point( _bin_width*(i-1), _hist_height - cvRound(_r_hist.at<float>(i-1)) ) , Point( _bin_width*(i), _hist_height - cvRound(_r_hist.at<float>(i)) ), Scalar( 0, 0, 255), 2, 8, 0  );
     }
    cv::namedWindow("hist", CV_WINDOW_AUTOSIZE);
    cv::imshow("hist", _hist_img);
    cv::waitKey(0);
}


void calcBackProjection( const std::vector<cv::Mat>& _in_images )
{
    cv::VideoCapture camera( 1 );
    cv::Mat frame;
    cv::Mat _in_image1;
    cv::Mat _dest_img;
    cv::Mat _hsv_img, _hsv_dest_img;
    cv::Mat _hue_ch, _hue_dest_ch;
    cv::MatND _hist;
    //set number of bins
    int _bins = 30;
    int _hist_sizes = 128;
    float _hue_range[] = {0, 180};
    const float* _ranges = { _hue_range };
    int ch[] = { 0, 0};
    for( int i = 0; i < _in_images.size(); i++ )
    {
            //_in_image1 = _in_images[i]; 
	    cv::cvtColor( _in_images[i], _hsv_img, CV_BGR2HSV);
	    //extract and use only the hue value
	    _hue_ch.create( _hsv_img.size(), _hsv_img.depth() );
	    cv::mixChannels( &_hsv_img, 1, &_hue_ch, 1, ch, 1);
	    /// Get the Histogram and normalize it(accumulate)
	    cv::calcHist( &_hue_ch, 1, 0, cv::Mat(), _hist, 1, &_hist_sizes, &_ranges, true, true );
	    cv::normalize( _hist, _hist, 0, 255, NORM_MINMAX, -1, cv::Mat() );
    }

    /// Draw the histogram
    cv::MatND _hist_img;
    int w = 400; int h = 400;
    int _bin_w = cvRound( (double) w / _hist_sizes );
    _hist_img = cv::Mat::zeros( w, h, CV_8UC3 );

    for( int i = 0; i < _bins; i ++ )
    {
         cv::rectangle( _hist_img, Point( i*_bin_w, h ), Point( (i+1)*_bin_w, h - cvRound( _hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 ); }

    cv::namedWindow("hist", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("BackProj", CV_WINDOW_AUTOSIZE);
    cv::imshow("hist", _hist_img);
    //Get Backprojection
    cv::MatND _backproj;
    while( true )
    {
        // Get the next frame.
        camera.grab();
        camera.retrieve(frame);
        _dest_img = frame;
        cv::cvtColor( _dest_img, _hsv_dest_img, CV_BGR2HSV);
        _hue_dest_ch.create( _hsv_dest_img.size(), _hsv_dest_img.depth() );
        cv::mixChannels( &_hsv_dest_img, 1, &_hue_dest_ch, 1, ch, 1);
        cv::calcBackProject( &_hue_dest_ch, 1, 0, _hist, _backproj, &_ranges, 1, true );
       /// Draw the backproj
       cv::imshow( "BackProj", _backproj );
       detectEdges( _backproj );
       if( cv::waitKey(10)>10 ) break;
    }

}

void detectEdges( cv::Mat& _in_image )
{
    cv::Mat _dst_image;
    cv::Mat _dst;
    int _threshold_low = 10;
    int _ratio = 3;
    float _sigma = 1;
    //blur to reduce noise
    cv::GaussianBlur(_in_image, _dst_image, Size(5,5), _sigma, 0, BORDER_DEFAULT);
    //use canny
    cv::Canny(_dst_image, _dst_image, _threshold_low, _ratio*_threshold_low);
    //transform angles to pixels densities using canny's output as a mask
    _dst = Scalar::all(0);
    _in_image.copyTo( _dst, _dst_image);
    cv::namedWindow("Edges", CV_WINDOW_AUTOSIZE);
    cv:imshow( "Edges", _dst);
    detectBlobs( _dst );
}

void detectBlobs( cv::Mat& _in_image )
{
    //define parameters
    cv::SimpleBlobDetector::Params params;
    params.minDistBetweenBlobs = 50.0f;
    params.filterByColor = false;
    params.filterByArea = true;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity= false;
    params.minArea = 20.0f;
    params.maxArea = 500.0f;
    //create the detector with the above params
    cv::Ptr<cv::FeatureDetector> _blob_detector = new cv::SimpleBlobDetector(params);
    _blob_detector->create("SimpleBlob");
    //detect and extract coordinates of keypoints
    vector<cv::KeyPoint> _keypoints;
    _blob_detector->detect(_in_image, _keypoints);
    for (int i=0; i<_keypoints.size(); i++)
    {
        float _x = _keypoints[i].pt.x;
        float _y = _keypoints[i].pt.y;
        printf( "X: %f, Y: %f/n",_x, _y );
    }

}  
