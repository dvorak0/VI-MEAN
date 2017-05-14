#ifndef MOTION_ESTIMATOR_H
#define MOTION_ESTIMATOR_H

#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>

class MotionEstimator
{
public:
    Matrix3d solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres);
private:
    double testTriangulation(const vector<cv::Point2f> &l,
                             const vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};

#endif
