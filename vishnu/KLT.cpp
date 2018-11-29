////
//// Created by sicong on 08/11/18.
////
//
//#include <iostream>
//#include <fstream>
//#include <list>
//#include <vector>
//#include <chrono>
//using namespace std;
//
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/video/tracking.hpp>
//
//using namespace cv;
//int main( int argc, char** argv )
//{
//
//    if ( argc != 3 )
//    {
//        cout<<"usage: feature_extraction img1 img2"<<endl;
//        return 1;
//    }
//    //-- Read two images
//    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
//    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
//
//    list< cv::Point2f > keypoints;
//    vector<cv::KeyPoint> kps;
//
//    std::string detectorType = "Feature2D.BRISK";
//    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
//	detector->set("thres", 100);
//
//
//    detector->detect( img_1, kps );
//    for ( auto kp:kps )
//        keypoints.push_back( kp.pt );
//
//    vector<cv::Point2f> next_keypoints;
//    vector<cv::Point2f> prev_keypoints;
//    for ( auto kp:keypoints )
//        prev_keypoints.push_back(kp);
//    vector<unsigned char> status;
//    vector<float> error;
//    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
//    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
//    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
//
//    // visualize all  keypoints
//    hconcat(img_1,img_2,img_1);
//    for ( int i=0; i< prev_keypoints.size() ;i++)
//    {
//        cout<<(int)status[i]<<endl;
//        if(status[i] == 1)
//        {
//            Point pt;
//            pt.x =  next_keypoints[i].x + img_2.size[1];
//            pt.y =  next_keypoints[i].y;
//
//            line(img_1, prev_keypoints[i], pt, cv::Scalar(0,255,255));
//        }
//    }
//
//    cv::imshow("klt tracker", img_1);
//    cv::waitKey(0);
//
//    return 0;
//}


//
// Created by sicong on 08/11/18.
//

#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <math.h>
using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;

cv::Mat convert_to_homogenous(cv::Point2f cartesian){
    cv::Mat homogenous_mat(3,1,CV_64FC1);

    homogenous_mat.at<double>(0,0)=cartesian.x;
    homogenous_mat.at<double>(1,0)=cartesian.y;
    homogenous_mat.at<double>(2,0)=1;

    return homogenous_mat;
}

cv::Mat get_epipolar_lines(cv::Mat point,cv::Mat Fundamental)
{
   cv::Mat epipolar_line(1,3,CV_64FC1);
   epipolar_line=(Fundamental.t())*point;
   return epipolar_line;
}

cv::Matx33d Findfundamental(vector<cv::Point2f> prev_subset,vector<cv::Point2f> next_subset){
    cv::Matx33d F;
    cv::Mat homog_x_prime(3,1,CV_64FC1);
    cv::Mat homog_x(3,1,CV_64FC1);
    cv::Mat line(1,3,CV_64FC1);
    cv::Mat normalisation_mat(3,3,CV_64FC1);
    cv::Point2f curr_point;
    cv::Mat A_inter(3,3,CV_64FC1);
    cv::Mat A(prev_subset.size(),9,CV_64FC1);
    cv::Mat AtA(9,9,CV_64FC1);
    cv::Mat fprime_est(3,3,CV_64FC1);
    cv::Mat fprime(3,3,CV_64FC1);
    cv::Mat svdestw = cv::Mat::zeros(3,3,CV_64FC1);
    //fill the blank

    normalisation_mat.at<double>(0,0)=(2.0/640);
    normalisation_mat.at<double>(0,1)= 0;
    normalisation_mat.at<double>(0,2)=-1;
    normalisation_mat.at<double>(1,0)=0;
    normalisation_mat.at<double>(1,1)=(2.0/480);
    normalisation_mat.at<double>(1,2)=-1;
    normalisation_mat.at<double>(2,0)=0;
    normalisation_mat.at<double>(2,1)=0;
    normalisation_mat.at<double>(2,2)=1;

    //cout<<"sizeeeeeeeeeeeeeeee"<<prev_subset.size()<<endl;

  
    for (size_t i=0;i<prev_subset.size();i++)
    {
    	homog_x_prime=convert_to_homogenous(next_subset[i]); 

         
    	homog_x=convert_to_homogenous(prev_subset[i]);

        
        homog_x_prime=normalisation_mat*homog_x_prime;
        homog_x=normalisation_mat*homog_x;


        //cout<<"Matrix after norm homog_x_prime"<<homog_x_prime<<endl;
        //cout<<"Matrix  after norm homog_x"<<homog_x<<endl;

    	A_inter = homog_x_prime*homog_x.t();

    	A.at<double>(i,0)=A_inter.at<double>(0,0);
    	A.at<double>(i,1)=A_inter.at<double>(0,1);
    	A.at<double>(i,2)=A_inter.at<double>(0,2);
    	A.at<double>(i,3)=A_inter.at<double>(1,0);
    	A.at<double>(i,4)=A_inter.at<double>(1,1);
    	A.at<double>(i,5)=A_inter.at<double>(1,2);
    	A.at<double>(i,6)=A_inter.at<double>(2,0);
    	A.at<double>(i,7)=A_inter.at<double>(2,1);
    	A.at<double>(i,8)=A_inter.at<double>(2,2);


    // cout<<"iterations"<<i<<endl;
    }

    cout<<"Matrix A"<<A.size()<<endl;
    cv::SVD svd(A);

    //cout<<"svd u"<<svd.u<<endl;
    //cout<<"svd vt"<<svd.vt<<endl;
    //cout<<"svd w"<<svd.w<<endl;

    fprime_est.at<double>(0,0)=svd.vt.at<double>(7,0);
    fprime_est.at<double>(0,1)=svd.vt.at<double>(7,1);
    fprime_est.at<double>(0,2)=svd.vt.at<double>(7,2);
    fprime_est.at<double>(1,0)=svd.vt.at<double>(7,3);
    fprime_est.at<double>(1,1)=svd.vt.at<double>(7,4);
    fprime_est.at<double>(1,2)=svd.vt.at<double>(7,5);
    fprime_est.at<double>(2,0)=svd.vt.at<double>(7,6);
    fprime_est.at<double>(2,1)=svd.vt.at<double>(7,7);
    fprime_est.at<double>(2,2)=svd.vt.at<double>(7,8);

    cv::SVD svdest(fprime_est);

    //cout<<"svdest w"<<svdest.w<<endl;
    //cout<<"svdest u"<<svdest.u<<endl;
    //cout<<"svdest vt"<<svdest.vt<<endl;

    svdestw.at<double>(0,0)=svdest.w.at<double>(0,0);
    svdestw.at<double>(1,1)=svdest.w.at<double>(1,0);

    fprime=svdest.u*svdestw*svdest.vt;

    //cout<<"prime "<<fprime<<endl;

    fprime = normalisation_mat.t()*fprime*normalisation_mat;

    //cout<<"check "<<homog_x_prime.t()*fprime*homog_x<<endl;

    F= fprime;

    //cout<<"Fundamental matrix "<<F<<endl;
    
    return F;
}
bool checkinlier(cv::Point2f prev_keypoint,cv::Point2f next_keypoint,cv::Matx33d Fcandidate,double d){
    //fill the blank
    cv::Mat homog_x_prime(3,1,CV_64FC1);
    cv::Mat homog_x(3,1,CV_64FC1);
    cv::Mat line(1,3,CV_64FC1);
    cv::Mat Fcand(3,3,CV_64FC1);

    homog_x_prime.at<double>(0,0)=next_keypoint.x;
    homog_x_prime.at<double>(1,0)=next_keypoint.y;
    homog_x_prime.at<double>(2,0)=1; 

    //cout<<"Matrix homog_x_prime"<<homog_x_prime<<endl;

    homog_x.at<double>(0,0)=prev_keypoint.x;
    homog_x.at<double>(1,0)=prev_keypoint.y;
    homog_x.at<double>(2,0)=1;

    //cout<<"Matrix homog_x"<<homog_x<<endl;

    Fcand = Mat(Fcandidate);

    line=(Fcand.t())*homog_x_prime;

    cout<<"epipolar line"<<line<<endl;

    float a = line.at<double>(0,0);
    float b = line.at<double>(1,0);
    float c = line.at<double>(2,0);
    float u = homog_x.at<double>(0,0);
    float v = homog_x.at<double>(1,0);
    float dist = abs(a*u+b*v+c)/sqrt(a*a+b*b);
    cout<<"dist"<<dist<<endl;
    cout<<"u"<<u<<endl;
    cout<<"v"<<v<<endl;

    if (dist <= d)
    {
        return true;
    }
    else
    {
        return false;
    }
}

int main( int argc, char** argv )
{

    srand ( time(NULL) );

    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- Read two images
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    list< cv::Point2f > keypoints;
    vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    Ptr<FeatureDetector>detector = Algorithm::create<FeatureDetector>(detectorType);
    detector->set("thres", 100);


    detector->detect( img_1, kps );
    for ( auto kp:kps )
        keypoints.push_back( kp.pt );

    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( auto kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK( img_1, img_2, prev_keypoints, next_keypoints, status, error );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    vector<cv::Point2f> kps_prev,kps_next;
    kps_prev.clear();
    kps_next.clear();
    for(size_t i=0;i<prev_keypoints.size();i++)
    {
        if(status[i] == 1)
        {
            kps_prev.push_back(prev_keypoints[i]);
            kps_next.push_back(next_keypoints[i]);
        }
    }


    // p Probability that at least one valid set of inliers is chosen
    // d Tolerated distance from the model for inliers
    // e Assumed outlier percent in data set.
    double p = 0.99;
    double d = 1.5f;
    double e = 0.2;

    int niter = static_cast<int>(std::ceil(std::log(1.0-p)/std::log(1.0-std::pow(1.0-e,8))));
    Mat Fundamental;
    cv::Matx33d F,Fcandidate;
    int bestinliers = -1;
    vector<cv::Point2f> prev_subset,next_subset;
    int matches = kps_prev.size();
    prev_subset.clear();
    next_subset.clear();

    for(int i=0;i<niter;i++){
        // step1: randomly sample 8 matches for 8pt algorithm
        unordered_set<int> rand_util;
        while(rand_util.size()<8)
        {
            int randi = rand() % matches;
            rand_util.insert(randi);
        }
        vector<int> random_indices (rand_util.begin(),rand_util.end());
        for(size_t j = 0;j<rand_util.size();j++){
            prev_subset.push_back(kps_prev[random_indices[j]]);
            next_subset.push_back(kps_next[random_indices[j]]);
        }
        // step2: perform 8pt algorithm, get candidate F
        Fcandidate = Findfundamental(prev_subset,next_subset);
        // step3: Evaluate inliers, decide if we need to update the best solution
        int inliers = 0;
        for(size_t j=0;j<kps_prev.size();j++){
            if(checkinlier(kps_prev[j],kps_next[j],Fcandidate,d))
                inliers++;
        }
        cout<<"No of inliers "<< inliers <<endl;
        if(inliers > bestinliers)
        {
            F = Fcandidate;
            bestinliers = inliers;
            cout<<"Best inliers "<< bestinliers <<endl;
        }
        prev_subset.clear();
        next_subset.clear();
    }

    // step4: After we finish all the iterations, use the inliers of the best model to compute Fundamental matrix again.

    for(size_t j=0;j<kps_prev.size();j++){
        if(checkinlier(kps_prev[j],kps_next[j],F,d))
        {
            prev_subset.push_back(kps_prev[j]);
            next_subset.push_back(kps_next[j]);
        }

    }
    cout<<"Get Best Fundamental matrix is \n"<<endl;
    F = Findfundamental(prev_subset,next_subset);

    cout<<"Best Fundamental matrix is \n"<<F<<endl;

    

   //visualize epipolar lines
   hconcat(img_1,img_2,img_1);
   for ( size_t i=0; i< next_subset.size() ;i++)
   {
       Point pt1;
       Point pt2;
       Point pt3;

       cv::Mat next_mat = convert_to_homogenous(next_subset[i]);
       cv::Mat prev_mat = convert_to_homogenous(prev_subset[i]);

       cv::Mat epipolar_line=get_epipolar_lines(next_mat,Mat(F));

       pt1.x = 0;
       pt1.y = -epipolar_line.at<double>(0,2)/epipolar_line.at<double>(0,1);

       pt2.x = img_2.cols;
       pt2.y = -(epipolar_line.at<double>(0,2)+epipolar_line.at<double>(0,0)*img_2.cols)/epipolar_line.at<double>(0,1);

       line(img_1, pt1, pt2, cv::Scalar(0,255,255));
       circle(img_1, prev_subset[i], 5, cv::Scalar(0, 0, 255), -1);

       epipolar_line=get_epipolar_lines(prev_mat,Mat(F).t());
       pt1.x = img_2.cols;
       pt1.y = -epipolar_line.at<double>(0,2)/(epipolar_line.at<double>(0,1));

       pt2.x = img_1.cols;
       pt2.y = -(epipolar_line.at<double>(0,2)+epipolar_line.at<double>(0,0)*img_2.cols)/epipolar_line.at<double>(0,1);

       pt3.x = next_subset[i].x+img_2.cols;
       pt3.y = next_subset[i].y;

       line(img_1, pt1, pt2, cv::Scalar(255,255,0));
       circle(img_1, pt3, 5, cv::Scalar(0, 255, 0), -1);


    }
   

   cv::imshow("klt tracker", img_1);
   cv::waitKey(0);






    return 0;
}