#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <ctime>
#include <chrono>
#include <unordered_set>
#include <math.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>

cv::Mat convert_to_homogenous(cv::Point2f cartesian)
{
    cv::Mat homogenous_mat(3, 1, CV_64FC1);
    homogenous_mat.at<double>(0, 0) = cartesian.x;
    homogenous_mat.at<double>(1, 0) = cartesian.y;
    homogenous_mat.at<double>(2, 0) = 1;
    return homogenous_mat;
}

cv::Matx33d findFundamental(
    std::vector<cv::Point2f> prev_subset, std::vector<cv::Point2f> next_subset)
{
    cv::Matx33d F;
    cv::Mat homog_x_prime(3, 1, CV_64FC1);
    cv::Mat homog_x(3, 1, CV_64FC1);
    cv::Mat line(1, 3, CV_64FC1);
    cv::Mat normalisation_mat(3, 3, CV_64FC1);
    cv::Point2f curr_point;
    cv::Mat A_inter(3, 3, CV_64FC1);
    cv::Mat A(prev_subset.size(), 9, CV_64FC1);
    cv::Mat AtA(9, 9, CV_64FC1);
    cv::Mat fprime_est(3, 3, CV_64FC1);
    cv::Mat fprime(3, 3, CV_64FC1);
    cv::Mat svdestw = cv::Mat::zeros(3, 3, CV_64FC1);

    normalisation_mat.at<double>(0, 0) = (2.0 / 640);
    normalisation_mat.at<double>(0, 1) = 0;
    normalisation_mat.at<double>(0, 2) = -1;
    normalisation_mat.at<double>(1, 0) = 0;
    normalisation_mat.at<double>(1, 1) = (2.0 / 480);
    normalisation_mat.at<double>(1, 2) = -1;
    normalisation_mat.at<double>(2, 0) = 0;
    normalisation_mat.at<double>(2, 1) = 0;
    normalisation_mat.at<double>(2, 2) = 1;
  
    for (size_t i = 0; i < prev_subset.size(); i++)
    {
        homog_x_prime=convert_to_homogenous(next_subset[i]);
        homog_x=convert_to_homogenous(prev_subset[i]); 
        homog_x_prime = normalisation_mat * homog_x_prime;
        homog_x = normalisation_mat * homog_x;

        A_inter = homog_x_prime * homog_x.t();

        A.at<double>(i, 0) = A_inter.at<double>(0, 0);
        A.at<double>(i, 1) = A_inter.at<double>(0, 1);
        A.at<double>(i, 2) = A_inter.at<double>(0, 2);
        A.at<double>(i, 3) = A_inter.at<double>(1, 0);
        A.at<double>(i, 4) = A_inter.at<double>(1, 1);
        A.at<double>(i, 5) = A_inter.at<double>(1, 2);
        A.at<double>(i, 6) = A_inter.at<double>(2, 0);
        A.at<double>(i, 7) = A_inter.at<double>(2, 1);
        A.at<double>(i, 8) = A_inter.at<double>(2, 2);
    }

    cv::SVD svd(A);

    fprime_est.at<double>(0, 0) = svd.vt.at<double>(7, 0);
    fprime_est.at<double>(0, 1) = svd.vt.at<double>(7, 1);
    fprime_est.at<double>(0, 2) = svd.vt.at<double>(7, 2);
    fprime_est.at<double>(1, 0) = svd.vt.at<double>(7, 3);
    fprime_est.at<double>(1, 1) = svd.vt.at<double>(7, 4);
    fprime_est.at<double>(1, 2) = svd.vt.at<double>(7, 5);
    fprime_est.at<double>(2, 0) = svd.vt.at<double>(7, 6);
    fprime_est.at<double>(2, 1) = svd.vt.at<double>(7, 7);
    fprime_est.at<double>(2, 2) = svd.vt.at<double>(7, 8);

    cv::SVD svdest(fprime_est);

    svdestw.at<double>(0, 0) = svdest.w.at<double>(0, 0);
    svdestw.at<double>(1, 1) = svdest.w.at<double>(1, 0);

    fprime = svdest.u * svdestw * svdest.vt;

    fprime = normalisation_mat.t() * fprime * normalisation_mat;

    F = fprime;

    return F;
}

bool checkInlier(
    cv::Point2f prev_keypoint, cv::Point2f next_keypoint,
    cv::Matx33d candidate, double threshold)
{
    // Convert candidate to cv::Mat.
    cv::Mat candidate_mat = cv::Mat(candidate);

    // Prepare data in homogeneous coordinates.
    cv::Mat prev_keypoint_h = cv::Mat(3, 1, CV_64FC1);
    cv::Mat next_keypoint_h = cv::Mat(3, 1, CV_64FC1);
    prev_keypoint_h.at<double>(0, 0) = prev_keypoint.x;
    prev_keypoint_h.at<double>(1, 0) = prev_keypoint.y;
    prev_keypoint_h.at<double>(2, 0) = 1.0;
    next_keypoint_h.at<double>(0, 0) = next_keypoint.x;
    next_keypoint_h.at<double>(1, 0) = next_keypoint.y;
    next_keypoint_h.at<double>(2, 0) = 1.0;

    // Compute epipolar line.
    cv::Mat epipolar_line = candidate_mat.t() * next_keypoint_h;

    // Compute the distance between the point and the line.
    double a = epipolar_line.at<double>(0, 0);
    double b = epipolar_line.at<double>(1, 0);
    double c = epipolar_line.at<double>(2, 0);
    double u = prev_keypoint_h.at<double>(0, 0);
    double v = prev_keypoint_h.at<double>(1, 0);
    double distance = std::abs(a * u + b * v + c) / std::sqrt(a * a + b * b);

    return (distance < threshold);
}

void visulize(
    std::vector<cv::Point2f> prev_keypoints, 
    std::vector<cv::Point2f> next_keypoints, 
    cv::Matx33d f, double threshold,
    cv::Mat img_out)
{
    double sum = 0;
    cv::Mat fundamental = cv::Mat(f);

    for (size_t i = 0; i < prev_keypoints.size(); i++) 
    {
        cv::Point2f prev_keypoint = prev_keypoints[i];
        cv::circle(
            img_out, prev_keypoint, 5, cv::Scalar(0, 0, 255), -1, CV_AA);
        
        cv::Point2f next_keypoint = next_keypoints[i];
        cv::Point2f next_keypoint_shift = next_keypoint;
        next_keypoint_shift.x += img_out.cols / 2.0;
        cv::circle(
            img_out, next_keypoint_shift, 5, cv::Scalar(255, 0, 0), -1, CV_AA);
        
        // Draw correspondences.
        cv::line(img_out, prev_keypoint, next_keypoint_shift, 
            cv::Scalar(0, 255, 255), 1, CV_AA);

        // Prepare data.
        cv::Mat epipolar_line;
        double a, b, c, u, v, x, y; 
        cv::Point2f p1, p2; 
        cv::Mat prev_keypoint_h = cv::Mat(3, 1, CV_64FC1);
        cv::Mat next_keypoint_h = cv::Mat(3, 1, CV_64FC1);
        prev_keypoint_h.at<double>(0, 0) = prev_keypoint.x;
        prev_keypoint_h.at<double>(1, 0) = prev_keypoint.y;
        prev_keypoint_h.at<double>(2, 0) = 1.0;
        next_keypoint_h.at<double>(0, 0) = next_keypoint.x;
        next_keypoint_h.at<double>(1, 0) = next_keypoint.y;
        next_keypoint_h.at<double>(2, 0) = 1.0;

        // Left Image =======================================================//
        epipolar_line = fundamental.t() * next_keypoint_h;

        // Compute the distance between the point and the line.
        a = epipolar_line.at<double>(0, 0);
        b = epipolar_line.at<double>(1, 0);
        c = epipolar_line.at<double>(2, 0);
        u = next_keypoint_h.at<double>(0, 0);
        v = prev_keypoint_h.at<double>(1, 0);
        sum += std::abs(a * u + b * v + c) / std::sqrt(a * a + b * b);
      
        x = 0;
        y = -(a * x + c) / b;
        p1 = cv::Point2f(x, y);

        x = img_out.cols / 2.0;
        y = -(a * x + c) / b;
        p2 = cv::Point2f(x, y);

        cv::line(img_out, p1, p2, cv::Scalar(0, 255, 0), 1, CV_AA);

        // Right Image ======================================================//
        epipolar_line = fundamental * prev_keypoint_h;

        // Compute the distance between the point and the line.
        a = epipolar_line.at<double>(0, 0);
        b = epipolar_line.at<double>(1, 0);
        c = epipolar_line.at<double>(2, 0);
        u = next_keypoint_h.at<double>(0, 0);
        v = next_keypoint_h.at<double>(1, 0);
        sum += std::abs(a * u + b * v + c) / std::sqrt(a * a + b * b);
      
        x = 0;
        y = -(a * x + c) / b;
        x += img_out.cols / 2.0;
        p1 = cv::Point2f(x, y);

        x = img_out.cols / 2.0;
        y = -(a * x + c) / b;
        x += img_out.cols / 2.0;
        p2 = cv::Point2f(x, y);

        cv::line(img_out, p1, p2, cv::Scalar(0, 255, 0), 1, CV_AA);
    }

    // Display epipole.
    // cv::Mat zero = cv::Mat(3, 1, CV_64FC1);
    // zero.at<double>(0, 0) = 0;
    // zero.at<double>(1, 0) = 0;
    // zero.at<double>(2, 0) = 0;
    // cv::Mat result;
    // cv::solve(fundamental, zero, result);
    // std::cout << "Result: " << result << std::endl;

    sum /= 2.0;
    cv::imwrite("out.png", img_out);
    cv::imshow("Visual SLAM", img_out);
    std::cout << "Average Error: " << sum / prev_keypoints.size() << "\n";
    std::cout << "Fundamental Matrix: \n" << fundamental << std::endl;
    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    if (argc != 3)
    {
        std::cout << "Usage: feature_extraction img1 img2" << std::endl;
        return 1;
    }

    // Read two images.
    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_out;
    cv::hconcat(img_1, img_2, img_out);

    std::list<cv::Point2f> keypoints;
    std::vector<cv::KeyPoint> kps;

    std::string detectorType = "Feature2D.BRISK";
    cv::Ptr<cv::FeatureDetector>detector = 
        cv::Algorithm::create<cv::FeatureDetector>(detectorType);
    detector->set("thres", 100);

    detector->detect(img_1, kps);
    for (auto kp : kps)
        keypoints.push_back(kp.pt);

    std::vector<cv::Point2f> next_keypoints;
    std::vector<cv::Point2f> prev_keypoints;
    for (auto kp : keypoints)
        prev_keypoints.push_back(kp);
    std::vector<unsigned char> status;
    std::vector<float> error;
    std::chrono::steady_clock::time_point t1;
    t1 = std::chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(
        img_1, img_2, prev_keypoints, next_keypoints, status, error);
    std::chrono::steady_clock::time_point t2;
    t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used;
    time_used = 
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "LK Flow Time：" << time_used.count() << " seconds." << "\n";

    std::vector<cv::Point2f> kps_prev, kps_next;
    kps_prev.clear();
    kps_next.clear();
    for (size_t i = 0; i < prev_keypoints.size(); i++)
    {
        if (status[i] == 1)
        {
            kps_prev.push_back(prev_keypoints[i]);
            kps_next.push_back(next_keypoints[i]);
        }
    }

    double p = 0.99; // p Probability that at least one valid set is chosen.
    double d = 1.5f; // d Tolerated distance from the model for inliers.
    double e = 0.2;  // e Assumed outlier percent in data set.

    int niter = static_cast<int>(
        std::ceil(std::log(1.0 - p) / std::log(1.0 - std::pow(1.0 - e, 8))));
    cv::Matx33d fundamental, candidate;
    std::vector<cv::Point2f> prev_subset, next_subset, inliers, best_inliers;
    int matches = kps_prev.size();
    prev_subset.clear();
    next_subset.clear();
    inliers.clear();
    best_inliers.clear();

    for (int i = 0; i < niter; i++)
    {
        // Step 1: Randomly sample 8 matches for 8pt algorithm.
        std::unordered_set<int> rand_util;
        while (rand_util.size() < 8)
        {
            int randi = rand() % matches;
            rand_util.insert(randi);
        }
        std::vector<int> random_indices(rand_util.begin(),rand_util.end());
        for (size_t j = 0; j < rand_util.size(); j++)
        {
            prev_subset.push_back(kps_prev[random_indices[j]]);
            next_subset.push_back(kps_next[random_indices[j]]);
        }

        // Step 2: Perform 8pt algorithm, get candidate fundamental matrix.
        candidate = findFundamental(prev_subset,next_subset);

        // Step 3: Evaluate inliers, decide if we need to update the solution.
        for (size_t j = 0; j < kps_prev.size(); j++)
        {
            if (checkInlier(
                kps_prev[j], kps_next[j], candidate, d))
                inliers.push_back(next_keypoints[j]);
        }
        if (inliers.size() > best_inliers.size())
        {
            fundamental = candidate;
            best_inliers = inliers;
        }
        prev_subset.clear();
        next_subset.clear();
        inliers.clear();
    }

    // Step 4: After we finish all the iterations, use the inliers of the best
    // model to compute Fundamental matrix again.
    for (size_t i = 0; i < kps_prev.size(); i++)
    {
        if (checkInlier(kps_prev[i], kps_next[i], fundamental, d))
        {
            prev_subset.push_back(kps_prev[i]);
            next_subset.push_back(kps_next[i]);
        }
    }
    std::cout << "Number of Inlier: " << prev_subset.size();
    std::cout << "/" << prev_keypoints.size() << "\n";

    fundamental = cv::findFundamentalMat(prev_keypoints, next_keypoints);
    fundamental = findFundamental(prev_subset, next_subset);
    visulize(prev_keypoints, next_keypoints, fundamental, d, img_out);

    // OpenCV ===============================================================//
    fundamental = cv::findFundamentalMat(prev_keypoints, next_keypoints);
    std::vector<cv::Vec3f> epilines1, epilines2;
    cv::computeCorrespondEpilines(prev_subset, 1, fundamental, epilines1);
    cv::computeCorrespondEpilines(next_subset, 2, fundamental, epilines2);
    for(size_t i = 0; i < epilines1.size(); i++)
    {
        cv::Scalar color(255, 255, 255);
     
        cv::line(img_out,
          cv::Point(img_1.cols + 0, 
            -epilines1[i][2] / epilines1[i][1]),
          cv::Point(img_1.cols * 2, 
            -(epilines1[i][2] + epilines1[i][0] * img_1.cols) / 
            epilines1[i][1]), color, 1, CV_AA);
     
        cv::line(img_out,
          cv::Point(0, -epilines2[i][2] / epilines2[i][1]),
          cv::Point(img_1.cols, 
            -(epilines2[i][2] + epilines2[i][0] * img_1.cols) / 
            epilines2[i][1]), color, 1, CV_AA);
    }
    cv::imwrite("out_opencv.png", img_out);
    cv::imshow("OpenCV Results", img_out);
    cv::waitKey(0);

    return 0;
}
