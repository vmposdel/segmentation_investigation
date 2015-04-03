/*********************************************************************
*
* Software License Agreement (BSD License)
*
* Copyright (c) 2014, P.A.N.D.O.R.A. Team.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:
*
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above
* copyright notice, this list of conditions and the following
* disclaimer in the documentation and/or other materials provided
* with the distribution.
* * Neither the name of the P.A.N.D.O.R.A. Team nor the names of its
* contributors may be used to endorse or promote products derived
* from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
* LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*
* Authors: Vasilis Bosdelekidis
*********************************************************************/

#include "include/depth_experiments.h"

using namespace std;
using namespace cv;

Mat sigma;
vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
int thresh = 30;
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
double hugeContourThresh = 25000;
int lowerContourNumberToTestHuge = 4;
double smallContourThresh = 2500;
double tinyContourThresh = 800;
float intensityThresh = 200.0;
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
int neighborThresh = 100;
int rayDiffThresh = 0.5 * validateThresh;
float neighborWeight = 0.8;
float rectDiffThresh = 0.8;
int neighborValueThresh = 30;

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

bool validateContours(int ci, std::vector<Point2f>* mcv, std::vector<int>* contourHeight, std::vector<int>* contourWidth)
{
    if(!secondPass)
    {
        int sumWhites = 0;
        if(cv::contourArea(contours[ci]) < hugeContourThresh && cv::contourArea(contours[ci]) > tinyContourThresh)
        {
            for(int i = (*mcv)[ci].y - validateThresh; i < (*mcv)[ci].y + validateThresh; i ++)
                for(int j = (*mcv)[ci].x - validateThresh; j < (*mcv)[ci].x + validateThresh; j ++)
                    if(j >= 0 && j <= sigma.cols && i >= 0 && i <= sigma.rows )
                    {
                        if((int)sigma.at<uchar>(j, i) == 255)
                        {
                            sumWhites ++;
                        }
                    }
            bool horizontalSerie = true;
            int horizontalMaxRay = 0;
            for(int i = (*mcv)[ci].y - validateThresh; i < (*mcv)[ci].y + validateThresh; i ++)
            {
                int horizontalSum = 0;
                horizontalSerie = true;
                for(int j = (*mcv)[ci].x - validateThresh; j < (*mcv)[ci].x + validateThresh; j ++)
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
            for(int i = (*mcv)[ci].x - validateThresh; i < (*mcv)[ci].x + validateThresh; i ++)
            {
                int verticalSum = 0;
                verticalSerie = true;
                for(int j = (*mcv)[ci].y - validateThresh; j < (*mcv)[ci].y + validateThresh; j ++)
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

            //if(sumWhites > lowerWhitesThresh && sumWhites < higherWhitesThresh && ((!horizontalSerie && !verticalSerie) || abs(horizontalMaxRay - verticalMaxRay) < rayDiffThresh))
            //    return true;
            //else
            //    return false;
            return true;
        }
        else
            return false;
    }
    else
    {
        if(cv::contourArea(contours[ci]) > smallContourThresh)
        {
            std::vector<int> contoursToMerge;
            for(int i = 0; i < contours.size(); i ++)
            {
                if(i != ci)
                    if(realContours.at(i) && (abs((*mcv)[ci].x - (*mcv)[i].x) < neighborThresh) && (abs((*mcv)[ci].y - (*mcv)[i].y) < neighborThresh) && (abs((int)image.at<uchar>((*mcv)[ci].x, (*mcv)[ci].y) - (int)image.at<uchar>((*mcv)[i].x, (*mcv)[i].y)) < neighborValueThresh))
                        contoursToMerge.push_back(i);
            }
            int newIndX = (*mcv)[ci].x;
            int newIndY = (*mcv)[ci].y;
            int contoursCounter = contoursToMerge.size();
            for(int i = 0; i < contoursToMerge.size(); i ++)
            {
                //if(cv::contourArea(contours[contoursToMerge[i]]) > cv::contourArea(contours[ci]))
                //{
                //    cout << "Merge" << imageNames.at(img) << "\n";
                //    (*mcv)[contoursToMerge[i]].x = ((*mcv)[contoursToMerge[i]].x * neighborWeight + (*mcv)[ci].x * (1 - neighborWeight));
                //    (*mcv)[contoursToMerge[i]].y = ((*mcv)[contoursToMerge[i]].y * neighborWeight + (*mcv)[ci].y * (1 - neighborWeight));
                //    (*contourHeight)[contoursToMerge[i]] = (*contourHeight)[contoursToMerge[i]] + (*contourHeight)[ci] + abs((*mcv)[ci].y - (*mcv)[contoursToMerge[i]].y);
                //    (*contourWidth)[contoursToMerge[i]] = (*contourWidth)[contoursToMerge[i]] + (*contourWidth)[ci] + abs((*mcv)[ci].x - (*mcv)[contoursToMerge[i]].x);
                //    return false;
                //}
                //else
                //{
                //    (*mcv)[ci].x = ((*mcv)[contoursToMerge[i]].x * (1 - neighborWeight) + (*mcv)[ci].x * neighborWeight);
                //    (*mcv)[ci].y = ((*mcv)[contoursToMerge[i]].y * (1 - neighborWeight) + (*mcv)[ci].y * neighborWeight);
                //    (*contourHeight)[ci] = (*contourHeight)[ci] + (*contourHeight)[contoursToMerge[i]] + abs((*mcv)[ci].y - (*mcv)[contoursToMerge[i]].y);
                //    (*contourWidth)[ci] = (*contourWidth)[ci] + (*contourWidth)[contoursToMerge[i]] + abs((*mcv)[ci].x - (*mcv)[contoursToMerge[i]].x);
                //    realContours.at(contoursToMerge[i]) = false;
                //}
                if(realContours.at(contoursToMerge[i]) && cv::contourArea(contours[contoursToMerge[i]]) > smallContourThresh)
                {
                    //realContours.at(contoursToMerge[i]) = validateContours(contoursToMerge[i], mcv, contourHeight, contourWidth); 
                    newIndX += (*mcv)[contoursToMerge[i]].x;
                    newIndY += (*mcv)[contoursToMerge[i]].y;
                    (*contourHeight)[ci] += (*contourHeight)[contoursToMerge[i]] + abs((*mcv)[ci].y - (*mcv)[contoursToMerge[i]].y);
                    (*contourWidth)[ci] += (*contourWidth)[contoursToMerge[i]] + abs((*mcv)[ci].x - (*mcv)[contoursToMerge[i]].x);
                    for(int j = 0; j < contours.size(); j ++)
                    {
                        if(j != ci && j != contoursToMerge[i])
                            if(realContours.at(j) && cv::contourArea(contours[j]) > smallContourThresh && (abs((*mcv)[contoursToMerge[i]].x - (*mcv)[j].x) < neighborThresh) && (abs((*mcv)[contoursToMerge[i]].y - (*mcv)[j].y) < neighborThresh) && (abs((int)image.at<uchar>((*mcv)[contoursToMerge[i]].x, (*mcv)[contoursToMerge[i]].y) - (int)image.at<uchar>((*mcv)[j].x, (*mcv)[j].y)) < neighborValueThresh))
                            {
                                newIndX += (*mcv)[j].x;
                                newIndY += (*mcv)[j].y;
                                contoursCounter ++;
                                realContours.at(j) = false;
                                (*contourHeight)[ci] += (*contourHeight)[j] + abs((*mcv)[ci].y - (*mcv)[j].y);
                                (*contourWidth)[ci] += (*contourWidth)[j] + abs((*mcv)[ci].y - (*mcv)[j].y);
                            }
                    }
                    realContours.at(contoursToMerge[i]) = false;
                    contoursCounter ++;
                }
            }
            if(contoursCounter > 0)
            {
                (*mcv)[ci].x = (int)(newIndX / contoursCounter);
                (*mcv)[ci].y = (int)(newIndY / contoursCounter);
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
    std::vector<vector<Point> > contours_poly( contours.size() );
    std::vector<cv::Rect> boundRect( contours.size() );
    std::vector<int> contourWidth(contours.size());
    std::vector<int> contourHeight(contours.size());
    //Mass center
    realContours.clear();
    for(int i = 0; i < contours.size(); i ++)
    {
        mu[i] = moments( contours[i], false );
        mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 );
        cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = cv::boundingRect( cv::Mat(contours_poly[i]) );
        contourHeight[i] = boundRect[i].height;
        contourWidth[i] = boundRect[i].width;
        realContours.push_back(validateContours(i, &mc, &contourHeight, &contourWidth));
    }
    secondPass = true;
    for(int i = 0; i < contours.size(); i ++)
    {
        if(realContours.at(i))
        {
            realContours.at(i) = validateContours(i, &mc, &contourHeight, &contourWidth);
        }
        if(realContours.at(i))
            if(contourWidth[i] > contourHeight[i] + rectDiffThresh * contourHeight[i] || contourHeight[i] > contourWidth[i] + rectDiffThresh * contourWidth[i])
                realContours.at(i) = false;
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
        structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(24, 24));
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
