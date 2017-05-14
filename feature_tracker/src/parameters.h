#pragma once
#include <ros/ros.h>

//const int ROW = 512;
//const int COL = 640;
//const int FOCAL_LENGTH = 460;
const int ROW = 480;
const int COL = 752;
const int FOCAL_LENGTH = 368;
const int NUM_OF_CAM = 1;
#define GPU 0

extern std::string CALIB_DIR;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern double T_THRESHOLD;
extern bool SHOW_TRACK;
extern bool STEREO_TRACK;
extern bool USE_F;
extern bool USE_E;
extern bool EQUALIZE;

void readParameters(ros::NodeHandle &n);
