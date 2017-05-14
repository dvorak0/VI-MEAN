#include "parameters.h"

int MAX_FEATURE_CNT;
int NUM_OF_ITER;
double CALIB_THRESHOLD_TIC;
double CALIB_THRESHOLD_RIC;
double INIT_DEPTH;
double GRADIENT_THRESHOLD;
double FEATURE_THRESHOLD;
double MIN_PARALLAX;
double MIN_PARALLAX_POINT;
double ERROR_THRESHOLD;
bool SHOW_HISTOGRAM;
bool SHOW_GRAPH;
bool SHOW_HTML;
bool MULTI_THREAD;
double IMU_RATE;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<bool> RIC_OK, TIC_OK;
std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.70};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
bool COMPENSATE_ROTATION;

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
    MAX_FEATURE_CNT = readParam<int>(n, "max_feature_cnt");
    NUM_OF_ITER = readParam<int>(n, "num_of_iter");
    CALIB_THRESHOLD_RIC = readParam<double>(n, "calib_threshold_ric");
    CALIB_THRESHOLD_TIC = readParam<double>(n, "calib_threshold_tic");
    INIT_DEPTH = readParam<double>(n, "init_depth");
    GRADIENT_THRESHOLD = readParam<double>(n, "gradient_threshold") / FOCAL_LENGTH;
    FEATURE_THRESHOLD = readParam<double>(n, "feature_threshold") / FOCAL_LENGTH;
    MIN_PARALLAX = readParam<double>(n, "min_parallax") / FOCAL_LENGTH;
    MIN_PARALLAX_POINT = readParam<double>(n, "min_parallax_point") / FOCAL_LENGTH;
    ERROR_THRESHOLD = readParam<double>(n, "error_threshold");
    SHOW_HISTOGRAM = readParam<bool>(n, "show_histogram");
    SHOW_GRAPH = readParam<bool>(n, "show_graph");
    SHOW_HTML = readParam<bool>(n, "show_html");
    MULTI_THREAD = readParam<bool>(n, "multi_thread");

    IMU_RATE = readParam<double>(n, "imu_rate");
    ACC_N = readParam<double>(n, "acc_n");
    ACC_W = readParam<double>(n, "acc_w");
    GYR_N = readParam<double>(n, "gyr_n");
    GYR_W = readParam<double>(n, "gyr_w");
    BIAS_ACC_THRESHOLD = readParam<double>(n, "bias_acc_threshold");
    BIAS_GYR_THRESHOLD = readParam<double>(n, "bias_gyr_threshold");
    SOLVER_TIME = readParam<double>(n, "solver_time");
    COMPENSATE_ROTATION = readParam<bool>(n, "compensate_rotation");

    Eigen::Matrix3d tmp;
    tmp << -1, 0, 0,
        0, -1, 0,
        0, 0, 1;
    std::cout << Utility::R2ypr(tmp) << std::endl;

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        RIC_OK.push_back(readParam<bool>(n, std::string("ric_ok") + std::to_string(i)));
        if (RIC_OK.back())
        {
            RIC.push_back(Utility::ypr2R(Eigen::Vector3d(
                readParam<double>(n, std::string("ric_y") + std::to_string(i)),
                readParam<double>(n, std::string("ric_p") + std::to_string(i)),
                readParam<double>(n, std::string("ric_r") + std::to_string(i)))));
            std::cout << RIC[i] << std::endl;
        }
        TIC_OK.push_back(readParam<bool>(n, std::string("tic_ok") + std::to_string(i)));
        if (TIC_OK.back())
        {
            TIC.push_back(Eigen::Vector3d(
                readParam<double>(n, std::string("tic_x") + std::to_string(i)),
                readParam<double>(n, std::string("tic_y") + std::to_string(i)),
                readParam<double>(n, std::string("tic_z") + std::to_string(i))));
        }
    }
}
