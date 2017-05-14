#pragma once

#include "parameters.h"
#include "motion_estimator.h"
#include "marginalization.h"
#include "feature_manager.h"
#include "utility.h"
#include "tic_toc.h"
#include "imu_factor.h"

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>

#include <ceres/ceres.h>
#include "pose_local_parameterization.h"
#include "projection_factor.h"
#include "marginalization_factor.h"

#include <thread>
#include <pthread.h>
#include <syscall.h>
#include <sys/types.h>
#include <unordered_map>
#include <queue>

class SelfCalibrationEstimator
{
  public:
    //typedef EulerIntegration Integration_t;
    typedef MidpointIntegration Integration_t;
    //typedef RK4Integration Integration_t;
    typedef IMUFactor<Integration_t> IMUFactor_t;

    SelfCalibrationEstimator();

    void setIMUModel();

    // interface
    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Vector3d>>> &image, const std_msgs::Header &header,
                      const map<int, Vector3d> &debug_image);

    // internal
    void clearState();
    void changeState();

    void propagateIMU(Matrix<double, 7, 1> &imu_linear, Matrix3d &IMU_angular,
                      Matrix<double, 6, 6> &imu_cov, Matrix<double, 9, 9> &imu_cov_nl,
                      double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity, const Vector3d &b_a, const Vector3d &b_g);

    double solveCalibration(int camera_id);
    void solveOdometry();
    void solveOdometryLinear(MatrixXd &A, VectorXd &b);

    void setPrior();

    void marginalize();
    void marginalizeFront();
    void marginalizeBack();

    // from imu_3dm_gx4
    const double acc_density = 1.0e-3;
    const double gyr_density = 8.73e-5;
    const double update_rate = 200.0;

    const Matrix3d acc_cov = std::pow(acc_density * std::sqrt(update_rate), 2.0) * Matrix3d::Identity(); // 0.014
    const Matrix3d gyr_cov = std::pow(gyr_density * std::sqrt(update_rate), 2.0) * Matrix3d::Identity(); // 0.0012
    const Matrix2d pts_cov = (0.5 / FOCAL_LENGTH) * (0.5 / FOCAL_LENGTH) * Matrix2d::Identity();

    // used in nonlinear, fixed first state
    const double prior_p_std = 0.0001;
    const double prior_q_std = 0.01 / 180.0 * M_PI;

    // used in nonlinear week assumption
    // const double tic_std = .005;
    // const double ric_std = .3 / 180.0 * M_PI;

    // used in nonlinear strong assumption
    const double tic_std = 0.005;
    const double ric_std = 0.01 / 180.0 * M_PI;

    const Matrix3d gra_cov = 0.001 * 0.001 * Matrix3d::Identity();

    enum SolverFlag
    {
        CALIBRATION,
        LINEAR,
        NON_LINEAR
    };
    SolverFlag solver_flag;
    bool marginalization_flag;
    int g_cnt;
    Vector3d g_sum;
    Vector3d g;
    MatrixXd Ap[2], backup_A;
    VectorXd bp[2], backup_b;

    Matrix3d ric[NUM_OF_CAM];
    Vector3d ric_cov[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM], tic_cov[NUM_OF_CAM];

    Vector3d last_ric[NUM_OF_CAM], last_tic[NUM_OF_CAM];
    Vector3d feature_dist;

    Vector3d Ps[10 * (WINDOW_SIZE + 1)];
    Vector3d Vs[10 * (WINDOW_SIZE + 1)];
    Matrix3d Rs[10 * (WINDOW_SIZE + 1)];
    Vector3d Bas[10 * (WINDOW_SIZE + 1)];
    Vector3d Bgs[10 * (WINDOW_SIZE + 1)];

    Matrix3d Rc[NUM_OF_CAM][10 * (WINDOW_SIZE + 1)];
    Matrix3d Rc_g[NUM_OF_CAM][10 * (WINDOW_SIZE + 1)];

    std_msgs::Header Headers[10 * (WINDOW_SIZE + 1)];
    int use_cov[10 * (WINDOW_SIZE + 1)];

    std::vector<std::pair<int, std::vector<long long>>> outlier_info;

    IMUFactor_t *imu_factors[10 * (WINDOW_SIZE + 1)];
    bool first_imu;
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[10 * (WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[10 * (WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[10 * (WINDOW_SIZE + 1)];

    Matrix<double, 7, 1> IMU_linear[10 * (WINDOW_SIZE + 1)];
    Matrix3d IMU_angular[10 * (WINDOW_SIZE + 1)];

    Matrix<double, 6, 6> IMU_cov[10 * (WINDOW_SIZE + 1)];
    Matrix<double, 9, 9> IMU_cov_nl[10 * (WINDOW_SIZE + 1)];

    int frame_count;
    int inv_cnt;
    int sum_of_outlier, sum_of_diverge, sum_of_back, sum_of_front, sum_of_invalid;

    cv::Mat img_graph;

    FeatureManager f_manager;
    Marginalization margin;
    MotionEstimator m_estimator;

    bool is_valid, is_key;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_map;
    vector<Vector3d> key_poses;
    Vector3d init_P;
    Vector3d init_V;
    Matrix3d init_R;

    bool ready_mapping;

    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];
    double para_Feature[NUM_OF_F][SIZE_FEATURE];
    double para_Ex_Pose[NUM_OF_CAM][SIZE_POSE];

    ceres::Problem problem;
    ceres::LossFunction *loss_function;

    MarginalizationFactor *last_marginalization_factor;
    vector<double *> last_marginalization_parameter_blocks;

    void solve_ceres();
    void old2new();
    void new2old();
};
