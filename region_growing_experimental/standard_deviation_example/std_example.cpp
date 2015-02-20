#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat mat2gray(const Mat& src)
{
    Mat dst;
    normalize(src, dst, 0.0, 1.0, NORM_MINMAX);
    return dst;
}

int main()
{
    Mat image = imread("../../negative21.jpg", 0);
    if(!image.data)
    {
        cout << "Cannot open image \n";
        return 0;
    }

    Mat image32f;
    image.convertTo(image32f, CV_32F);

    Mat mu;
    blur(image32f, mu, Size(3, 3));

    Mat mu2;
    blur(image32f.mul(image32f), mu2, Size(3, 3));

    Mat sigma;
    cv::sqrt(mu2 - mu.mul(mu), sigma);

    //imshow("coke", mat2gray(image32f));
    //imshow("mu", mat2gray(mu));
    //imshow("sigma",mat2gray(sigma));
    //imwrite("../../negative21_std.jpg", sigma);
    //waitKey();
    int windowSize = 18;
    int toCheckThresh = 2000;
    float toGrowThresh = 2;
    int grow = 1;
    int windowsX = (int)((sigma.cols) / windowSize);
    int windowsY = (int)((sigma.rows) / windowSize);
    int max = -1;
    int maxx;
    int maxy;
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
                grow = 1;
                int tempsum = 0;
                int tempsum1 = sum;
                int tempStartY = j * windowSize;
                int tempStartX = i * windowSize;
                int tempEndY =  maxyWindow - 1;
                int tempEndX =  maxxWindow - 1;
                if(tempStartY - windowSize >=0)
                    tempStartY = tempStartY - windowSize;
                if(tempStartY + windowSize <= sigma.rows)
                    tempEndY = tempStartY + windowSize;
                if(tempStartX - windowSize >=0)
                    tempStartX = tempStartX - windowSize;
                if(tempStartX + windowSize <= sigma.cols)
                    tempEndX = tempStartX + windowSize;
                while(grow)
                {
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
                        grow = 0;
                    else
                    {
                        if(tempStartY - windowSize >=0)
                            tempStartY = tempStartY - windowSize;
                        if(tempEndY + windowSize <= sigma.rows)
                            tempEndY = tempEndY + windowSize;
                        if(tempStartX - windowSize >=0)
                            tempStartX = tempStartX - windowSize;
                        if(tempEndX + windowSize <= sigma.cols)
                            tempEndX = tempEndX + windowSize;
                        tempsum1 = tempsum;
                    }
                }
                if(tempsum1 > max)
                {
                    max = tempsum;
                    maxx = (int) ((tempEndX + tempStartX) / 2);
                    maxy = (int) ((tempEndY + tempStartY) / 2);
                }
            }
        }
    }
    cout << "Max = " << max << "\n";
    cout << "maxx = " << maxx << " maxy = " << maxy << "\n";

    return 0;
}
