#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include "parameters.h"

const int N = 128;

cv::Mat cpu_src{HEIGHT, WIDTH, CV_8U};
cv::Mat cpu_dst{HEIGHT, WIDTH, CV_8U};
cv::Mat cpu_disp{HEIGHT, WIDTH, CV_16S};
cv::Mat H = cv::Mat::eye(3, 3, CV_64F);

cv::gpu::GpuMat gpu_src{HEIGHT, WIDTH, CV_32F};
cv::gpu::GpuMat gpu_dst{HEIGHT, WIDTH, CV_32F};

TEST(perspective, cpu)
{
    for (int i = 0; i < N; i++)
        cv::warpPerspective(cpu_src, cpu_dst, H, cpu_src.size());
}

TEST(perspective, gpu)
{
    for (int i = 0; i < N; i++)
        cv::gpu::warpPerspective(gpu_src, gpu_dst, H, gpu_src.size());
}

TEST(sgbm, cpu)
{
    cv::StereoSGBM sgbm;
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = DEP_CNT;
    sgbm.fullDP = false;
    sgbm(cpu_src, cpu_dst, cpu_disp);
}

TEST(sgbm, gpu)
{
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
