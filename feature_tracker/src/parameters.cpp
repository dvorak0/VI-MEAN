#include "parameters.h"

std::string CALIB_DIR;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
double T_THRESHOLD;
bool SHOW_TRACK;
bool STEREO_TRACK;
bool USE_F;
bool USE_E;
bool EQUALIZE;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    CALIB_DIR = readParam<std::string>(n, "calib_dir");
    for (int i = 0; i < NUM_OF_CAM; i++)
        CAM_NAMES.push_back(readParam<std::string>(n, "cam_name" + std::to_string(i)));
    MAX_CNT = readParam<int>(n, "max_cnt");
    MIN_DIST = readParam<int>(n, "min_dist");
    FREQ = readParam<int>(n, "freq");
    WINDOW_SIZE = readParam<int>(n, "window_size");
    F_THRESHOLD = readParam<double>(n, "F_threshold");
    T_THRESHOLD = readParam<double>(n, "T_threshold");
    SHOW_TRACK = readParam<bool>(n, "show_track");
    STEREO_TRACK = readParam<bool>(n, "stereo_track");
    USE_F = readParam<bool>(n, "use_F");
    USE_E = readParam<bool>(n, "use_E");
    EQUALIZE = readParam<bool>(n, "equalize");
}
