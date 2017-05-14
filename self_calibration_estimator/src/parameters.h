#pragma once

#include <ros/ros.h>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility.h"

const int ROW = 640;
const int COL = 512;
const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;

//#define INV_DEP
//#define DEPTH_PRIOR
//#define GT

extern int MAX_FEATURE_CNT;
extern int NUM_OF_ITER;
extern double CALIB_THRESHOLD_TIC;
extern double CALIB_THRESHOLD_RIC;
extern double INIT_DEPTH;
extern double GRADIENT_THRESHOLD;
extern double FEATURE_THRESHOLD;
extern double MIN_PARALLAX;
extern double MIN_PARALLAX_POINT;
extern double ERROR_THRESHOLD;
extern bool SHOW_HISTOGRAM;
extern bool MULTI_THREAD;
extern bool SHOW_GRAPH;
extern bool SHOW_HTML;

extern double IMU_RATE;
extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern std::vector<bool> RIC_OK, TIC_OK;
extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern bool COMPENSATE_ROTATION;

void readParameters(ros::NodeHandle &n);

enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
