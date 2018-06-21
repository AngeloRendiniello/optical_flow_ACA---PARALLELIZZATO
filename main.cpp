#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <fstream>
#include <string>
#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

/* 2-D convolution */
int convolution(Mat filter_mask, Mat points)
{

    int result=0;


    for (int i = 0; i < points.rows; i++)
    {
        for (int j = 0; j < points.cols; j++)
        {
            result=result + (filter_mask.at<char>(i, j)*points.at<uchar>(i,j));
        }

    }
    return result;
}

Mat conv2(Mat imgin, Mat mask)
{
    int ma = imgin.rows;
    int na = imgin.cols;
    int mb = mask.rows;
    int nb = mask.cols;

    Mat points = Mat::zeros( mask.size(), imgin.type());

    cv::Mat results = Mat::zeros(max(ma-max(mb-1,0),0), max(na-max(nb-1,0),0), CV_32S);

	//Necessary to make convolution and not correlation
    flip(mask, mask, -1);

    //Pixels to point assignment
    for (int i = 0; i < ma - 1 ; i++)
    {
        for (int j = 0; j < na - 1 ; j++)
        {
            for (int x = 0; x < nb; x++)
            {
                for (int y = 0; y < nb; y++)
                {
                    points.at<char>(x,y) = imgin.at<char>(i + x, j+y);

                }

            }
            //Give to the central point the exact result of Convolution Method
            results.at<int>(i, j) = convolution(mask, points);
        }
    }
    return results;
}


void meshgrid(const cv::Range &xgv, const cv::Range &ygv, cv::Mat &X, cv::Mat &Y)
{
    std::vector<int> t_x, t_y;
    for(int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
    for(int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);

    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}


void LucasKanade(Mat img1, Mat img2, float reduction)
{

    int ww = 45;
    int w = (ww%2)+(ww/2);

    char mx[5] = { -1, 1, -1, 1 };
    cv::Mat maskx = cv::Mat(2, 2, CV_8SC1, mx);


    char my[4] = { -1, -1, 1, 1 };
    cv::Mat masky = cv::Mat(2, 2, CV_8SC1, my);

    Mat maskt = Mat::ones(2, 2, maskx.type());

    Mat Ix_m = conv2(img1, maskx); // partial on x
    Mat Iy_m = conv2(img1, masky); // partial on y
    Mat It_m = conv2(img1, maskt) + conv2(img2, -maskt); // partial on t

    Mat u = Mat::zeros(img1.size(), CV_32FC1);
    Mat v = Mat::zeros(img2.size(), CV_32FC1);

    Mat u_deci;
    Mat v_deci;
    Mat X_deci;
    Mat Y_deci;

    Mat Ix = Mat::zeros(w*2+1, w*2+1, Ix_m.type());
    Mat Iy = Mat::zeros(w*2+1, w*2+1, Iy_m.type());
    Mat It = Mat::zeros(w*2+1, w*2+1, It_m.type());

    Mat Ix_v = Mat::zeros(Ix.rows*Ix.cols, 1, Ix_m.type());
    Mat Iy_v = Mat::zeros(Iy.rows*Iy.cols, 1, Iy_m.type());
    Mat b = Mat::zeros(Iy.rows*Iy.cols, 1, It_m.type());
    Mat A = Mat::zeros(Ix_v.rows, 2, Ix_m.type());
    Mat nu = Mat::zeros(2,1, CV_32FC1);

    int ind=0;

	// within window ww * ww
    for(int i= w; i<Ix_m.rows-w; i++)
    {
        for(int j= w; j<Ix_m.cols-w; j++)
        {
            for(int k = i-w; k < i+w+1; k++ )
            {
                for(int l=j-w; l < j+w+1; l++)
                {
                    Ix.at<int>(k-(i-w),l-(j-w))=Ix_m.at<int>(k,l);
                    Iy.at<int>(k-(i-w),l-(j-w))=Iy_m.at<int>(k,l);
                    It.at<int>(k-(i-w),l-(j-w))=It_m.at<int>(k,l);

                }
            }
            //equivalent to Ix(:) and Iy(:) and b=-It(:)
            while(ind<Ix.rows*Ix.cols)
            {
                for(int k1=0; k1<Ix.cols; k1++)
                {
                    for(int k2=0; k2<Ix.rows; k2++)
                    {
                        Ix_v.at<int>(ind,0)=Ix.at<int>(k2,k1);
                        Iy_v.at<int>(ind,0)=Iy.at<int>(k2,k1);
                        b.at<int>(ind,0)= -It.at<int>(k2,k1);
                        ind++;
                    }
                }
            }

            hconcat(Ix_v, Iy_v, A);
            b.convertTo(b, CV_32FC1);

            A.convertTo(A, CV_32FC1);
            nu=A.inv(DECOMP_SVD)*b; //get velocity here

            u.at<float>(i,j)=nu.at<float>(0,0);
            v.at<float>(i,j)=nu.at<float>(1,0);
            ind=0;
            b.convertTo(b, It.type());
            A.convertTo(A, Ix.type());
        }
    }

    //Downsize v and u
    if(u.rows%10==0 && u.cols%10==0)
    {
        u_deci = Mat::ones(u.rows/10, u.cols/10, u.type());
        v_deci = Mat::ones(v.rows/10, v.cols/10, v.type());

    }
    if(u.rows%10!=0 && u.cols%10==0)
    {
        u_deci = Mat::ones(u.rows/10+1, u.cols/10, u.type());
        v_deci = Mat::ones(v.rows/10+1, v.cols/10, v.type());
    }
    if(u.rows%10==0 && u.cols%10!=0)
    {
        u_deci = Mat::ones(u.rows/10, u.cols/10+1, u.type());
        v_deci = Mat::ones(v.rows/10, v.cols/10+1, v.type());
    }
    if(u.rows%10!=0 && u.cols%10!=0)
    {
        u_deci = Mat::ones(u.rows/10+1, u.cols/10+1, u.type());
        v_deci = Mat::ones(v.rows/10+1, v.cols/10+1, v.type());
    }

    for(int i = 0; i<u.rows; i+=10)
    {
        for(int j = 0; j<u.cols; j+=10)
        {
            u_deci.at<float>(i/10, j/10)=u.at<float>(i,j);
            v_deci.at<float>(i/10, j/10)=v.at<float>(i,j);
        }
    }

    int m = img1.rows*(1/reduction);
    int n = img1.cols*(1/reduction);

    cv::Mat X, Y;
    meshgrid(cv::Range(1, n), cv::Range(1, m), X, Y);

    if(X.rows%(int)(10*(1/reduction))==0 && X.cols%(int)(10*(1/reduction))==0)
    {
        X_deci = Mat::ones(X.rows/(10*(1/reduction)), X.cols/(10*(1/reduction)), X.type());
        Y_deci = Mat::ones(Y.rows/(10*(1/reduction)), Y.cols/(10*(1/reduction)), Y.type());
    }
    if(X.rows%(int)(10*(1/reduction))!=0 && X.cols%(int)(10*(1/reduction))==0)
    {
        X_deci = Mat::ones(X.rows/(10*(1/reduction))+1, X.cols/(10*(1/reduction)), X.type());
        Y_deci = Mat::ones(Y.rows/(10*(1/reduction))+1, Y.cols/(10*(1/reduction)), Y.type());
    }
    if(X.rows%(int)(10*(1/reduction))==0 && X.cols%(int)(10*(1/reduction))!=0)
    {
        X_deci = Mat::ones(X.rows/(10*(1/reduction)), X.cols/(10*(1/reduction))+1, X.type());
        Y_deci = Mat::ones(Y.rows/(10*(1/reduction)), Y.cols/(10*(1/reduction))+1, Y.type());
    }
    if(X.rows%(int)(10*(1/reduction))!=0 && X.cols%(int)(10*(1/reduction))!=0)
    {
        X_deci = Mat::ones(X.rows/(10*(1/reduction))+1, X.cols/(10*(1/reduction))+1, X.type());
        Y_deci = Mat::ones(Y.rows/(10*(1/reduction))+1, Y.cols/(10*(1/reduction))+1, Y.type());

    }

    for(int i = 0; i<X.rows; i+=10*(1/reduction))
    {
        for(int j = 0; j<X.cols; j+=10*(1/reduction))
        {
            X_deci.at<float>(i/(10*(1/reduction)), j/(10*(1/reduction)))=X.at<float>(i,j);
            Y_deci.at<float>(i/(10*(1/reduction)), j/(10*(1/reduction)))=Y.at<float>(i,j);
        }
    }
	
	// output results written into files
    std::ofstream output1("X_deci.txt");
    for (int k=0; k<X_deci.rows; k++)
    {
        for (int l=0; l<X_deci.cols; l++)
        {
            output1 << X_deci.at<int>(k,l) << " "; 
        }
        output1 << "" << endl;
    }

    std::ofstream output2("Y_deci.txt");
    for (int k=0; k<Y_deci.rows; k++)
    {
        for (int l=0; l<Y_deci.cols; l++)
        {
            output2 << Y_deci.at<int>(k,l) << " "; 
        }
        output2 << "" << endl;
    }

    std::ofstream output3("u_deci.txt");
    for (int k=0; k<u_deci.rows; k++)
    {
        for (int l=0; l<u_deci.cols; l++)
        {
            output3 << u_deci.at<float>(k,l) << " "; 
        }
        output3 << "" << endl;
    }

    std::ofstream output4("v_deci.txt");
    for (int k=0; k<v_deci.rows; k++)
    {
        for (int l=0; l<v_deci.cols; l++)
        {
            output4 << v_deci.at<float>(k,l) << " "; 
        }
        output4 << "" << endl;
    }
}

int main()
{
    float reduction = 0.5;

    Mat img1 = cv::imread("img0.pgm",0);
    resize(img1, img1, Size(), reduction, reduction, CV_INTER_CUBIC);

    Mat img2 = cv::imread("img1.pgm",0);
    resize(img2, img2, Size(), reduction, reduction, CV_INTER_CUBIC);

    LucasKanade(img1, img2, reduction);

    return 0;
}

