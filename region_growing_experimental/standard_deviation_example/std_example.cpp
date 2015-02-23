#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sys/time.h>

using namespace std;
using namespace cv;

int debugShow = 1;

Mat mat2gray(const Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 1.0, NORM_MINMAX);
    return dst;
}

int main()
{

    Mat image = imread("../../negative43.jpg", 0);//cv::IMREAD_ANYDEPTH|cv::IMREAD_ANYCOLOR);
    Mat sigma;
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
    if(debugShow)
    {
        char* source_window = "Source";
        namedWindow( source_window, CV_WINDOW_AUTOSIZE );
        imshow(source_window, sigma);
    }

    //imwrite("../../negative43_std.jpg", sigma);
    waitKey(0);
    int windowSize = 18;
    int toCheckThresh = 2700;
    float toGrowThresh = 1.5;
    int grow = 1;
    int windowsX = (int)((sigma.cols) / windowSize);
    int windowsY = (int)((sigma.rows) / windowSize);
    int max = -1;
    int maxx = 0;
    int maxy = 0;
    for(int j = 0; j < windowsY; j ++)
    {
        for(int i = 0; i < windowsX; i++)
        {
            int y = j * windowSize; 
            int x = i * windowSize; 
            int maxyWindow = j * windowSize + windowSize; 
            int maxxWindow = i * windowSize + windowSize; 
            int sum = 0;
            for(y; y < maxyWindow; y ++)
            {
                for(x; x < maxxWindow; x ++)
                {
                    uchar value = sigma.at<uchar>(y,x);
                    sum += (int)value; 
                }
            }
            if(sum >= toCheckThresh)
            {
                int tempsum = 0;
                int tempsum1 = sum;
                int tempsum2 = 0;
                int tempMaxX = i * windowSize + windowSize;
                int tempMaxY = j * windowSize + windowSize;
                int tempMinX = i * windowSize;
                int tempMinY = j * windowSize;
                int neighborCoeff[8][2];
                int maxAtomic = sum;
                int maxAtomicX = (i * windowSize);
                int maxAtomicY = (j * windowSize);
                int tempmaxAtomic = sum;
                int tempmaxAtomicX = (i * windowSize);
                int tempmaxAtomicY = (j * windowSize);
                neighborCoeff[0][0] = 0;
                neighborCoeff[0][1] = 1;
                neighborCoeff[1][0] = 1;
                neighborCoeff[1][1] = 1;
                neighborCoeff[2][0] = 1;
                neighborCoeff[2][1] = 0;
                neighborCoeff[3][0] = 1;
                neighborCoeff[3][1] = -1;
                neighborCoeff[4][0] = 0;
                neighborCoeff[4][1] = -1;
                neighborCoeff[5][0] = -1;
                neighborCoeff[5][1] = -1;
                neighborCoeff[6][0] = -1;
                neighborCoeff[6][1] = 0;
                neighborCoeff[7][0] = -1;
                neighborCoeff[7][1] = 1;
                // for each neighboring window
                for(int s = 0; s < 8; s++)
                {
                    grow = 1;
                    int tempStartY = j * windowSize;
                    int tempStartX = i * windowSize;
                    int tempEndY =  tempStartY + windowSize;
                    int tempEndX =  tempStartX + windowSize;
                    while(grow)
                    {
                        if(neighborCoeff[s][0] >= 0)
                        {
                            if(tempEndY + neighborCoeff[s][0] * windowSize <= sigma.rows)
                            {
                                tempStartY = tempStartY + neighborCoeff[s][0] * windowSize;
                                tempEndY = tempEndY + neighborCoeff[s][0] * windowSize;
                            }
                        }
                        else
                        {
                            if(tempStartY + neighborCoeff[s][0] * windowSize >= 0)
                            {
                                tempStartY = tempStartY + neighborCoeff[s][0] * windowSize;
                                tempEndY = tempStartY;
                            }
                        }
                        if(neighborCoeff[s][1] >= 0)
                        {
                            if(tempEndX + neighborCoeff[s][1] * windowSize <= sigma.cols)
                            {
                                tempStartX = tempStartX + neighborCoeff[s][1] * windowSize;
                                tempEndX = tempEndX + neighborCoeff[s][1] * windowSize;
                            }
                        }
                        else
                        {
                            if(tempStartX + neighborCoeff[s][1] * windowSize >= 0)
                            {
                                tempStartX = tempStartX + neighborCoeff[s][1] * windowSize;
                                tempEndX = tempStartX;
                            }
                        }
                        tempsum = 0;
                        for(int tempY = tempStartY; tempY <= tempEndY; tempY ++)
                        {
                            for(int tempX = tempStartX; tempX < tempEndX; tempX ++)
                            {
                                //cout << tempY << ", " << tempX << "\n";
                                uchar value = sigma.at<uchar>(tempY, tempX);
                                tempsum += (int)value; 
                            }
                        }
                        //cout << tempsum << "\n";
                        if(tempsum < toGrowThresh * tempsum1)
                        {
                            grow = 0;
                            if(tempEndY > tempMaxY)
                                tempMaxY = tempEndY;
                            if(tempEndX > tempMaxX)
                                tempMaxX = tempEndX;
                            if(tempStartY < tempMinY)
                                tempMinY = tempStartY;
                            if(tempStartX < tempMinX)
                                tempMinX = tempStartX;
                            tempsum1 = sum;
                        }
                        else
                        {
                            if(tempsum > maxAtomic)
                            {
                                maxAtomicX = (int) ((tempEndX + tempStartX)/2);
                                maxAtomicY = (int) ((tempEndY + tempStartY)/2);
                                maxAtomic = tempsum;
                            }
                            tempsum1 = tempsum;
                            tempsum2 += tempsum;
                        }
                    }
                }
                if(tempsum2 > max)
                {
                    max = tempsum2;
                    maxx = maxAtomicX;
                    maxy = maxAtomicY;
                    tempmaxAtomicX = maxx;
                    tempmaxAtomicY = maxy;
                    tempmaxAtomic = maxAtomic;
                    //maxx = (int) ((tempMaxX + tempMinX) / 2);
                    //maxy = (int) ((tempMaxY + tempMinY) / 2);
                }
                else
                {
                    maxAtomic = tempmaxAtomic;
                    maxAtomicX = tempmaxAtomicX;
                    maxAtomicY = tempmaxAtomicY;
                }
            }
        }
    }
    cout << "Max = " << max << "\n";
    cout << "maxx = " << maxx << " maxy = " << maxy << "\n";

    return 0;
}
