#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <eigen3/Eigen/Dense>

/*
#include <opengv/relative_pose/modules/fivept_nister/modules.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/relative_pose/modules/main.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>
*/

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/EquidistantCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/ScaramuzzaCamera.h"

#include "parameters.h"
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

template <typename T1, typename T2>
void reduceVector(vector<T1> &v, vector<T2> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

template <typename T1, typename T2>
void reduceVector2(vector<T1> &v, vector<T2> status)
{
    int j = 0;
    for (auto i : status)
        v[j++] = v[i];
    v.resize(j);
}

class FeatureTracker
{
  public:
    FeatureTracker();

    void readImage(const cv::Mat &_img);

    void setMask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void rejectWithF();

    //void rejectWithE();

    vector<cv::Point2f> undistortedPoints();

    cv::Mat mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<int> ids;
    vector<int> track_cnt;
    camodocal::CameraPtr m_camera;

    static int n_id, img_cnt;
    static map<int, vector<pair<cv::Point2f, cv::Point2f>>> feature_history;
#if GPU
    cv::gpu::GpuMat g_forw_img, g_corners, g_mask;

    cv::gpu::GpuMat g_cur_img, g_cur_pts, g_forw_pts, g_status;

    cv::gpu::PyrLKOpticalFlow g_tracker;

    cv::gpu::GoodFeaturesToTrackDetector_GPU g_detector;

#endif
};
