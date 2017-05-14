#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <ros/console.h>

#include "parameters.h"
#include "calc_cost.h"
#include "tic_toc.h"

class StereoMapper
{
  public:
    StereoMapper();
    void initIntrinsic(const cv::Mat &K1, const cv::Mat &D1, const cv::Mat &K2, const cv::Mat &D2);

    void initReference(const cv::Mat &_img_l);

    void update(const cv::Mat &_img_r, const cv::Mat &R_l, const cv::Mat &T_l, const cv::Mat &R_r, const cv::Mat &T_r);

    void epipolar(double x, double y, double z);

    cv::Mat output();

    cv::gpu::GpuMat raw_img_l, raw_img_r;
    cv::gpu::GpuMat img_l, img_r, img_warp, img_diff;

    cv::gpu::GpuMat raw_cost, sgm_cost;
    cv::gpu::GpuMat dep;
    cv::gpu::GpuMat tmp;

    cv::gpu::GpuMat map1_l, map2_l;
    cv::gpu::GpuMat map1_r, map2_r;

    cv::Mat nK1, nK2;

    cv::Mat _map1_l, _map2_l;
    cv::Mat _map1_r, _map2_r;

    cv::Mat img_intensity;

#if BENCHMARK
    cv::Mat img_intensity_r;
#endif

    cv::Mat R, T;

    int measurement_cnt;
};
