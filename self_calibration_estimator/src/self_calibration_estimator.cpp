#include "self_calibration_estimator.h"

SelfCalibrationEstimator::SelfCalibrationEstimator()
    : solver_flag(CALIBRATION),
      g_cnt{0}, first_imu{false},
      frame_count{0}, inv_cnt{0},
      sum_of_outlier{0}, sum_of_diverge{0}, sum_of_back{0}, sum_of_front{0}, sum_of_invalid{0},
      img_graph{1000, 1050, CV_8UC3},
      f_manager{Rs},
      is_valid{true}, is_key{true}, ready_mapping{false},
      last_marginalization_factor{nullptr}
{
    ROS_INFO("init begins");

    clearState();

    img_graph = cv::Scalar(0, 0, 0);
    img_graph.colRange(500, 550) = cv::Scalar(0, 0, 0);

    if (SHOW_GRAPH)
        cv::namedWindow("graph", cv::WINDOW_NORMAL);

    ROS_INFO("init finished");

    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        problem.SetParameterBlockConstant(para_Ex_Pose[i]);
    }
    for (int i = 0; i < NUM_OF_F; i++)
    {
        problem.AddParameterBlock(para_Feature[i], SIZE_FEATURE);
    }
}

void SelfCalibrationEstimator::setIMUModel()
{
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
}

void SelfCalibrationEstimator::clearState()
{
    ROS_ERROR("clear state");
    for (int i = 0; i < 10 * (WINDOW_SIZE + 1); i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();

        use_cov[i] = 1;
        IMU_linear[i].setZero();
        IMU_angular[i].setIdentity();
        IMU_cov[i].setZero();
        IMU_cov_nl[i].setZero();

        imu_factors[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i].setIdentity();
    }
    g_sum.setZero();

    f_manager.clearState();
}

void SelfCalibrationEstimator::changeState()
{
    ROS_ERROR("change state");
    solver_flag = LINEAR;

    for (int i = 0; i <= frame_count; i++)
        printf("dt: %f\n", IMU_linear[i](6));

    frame_count--;
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + frame_count + 1 - WINDOW_SIZE;
        ROS_INFO("changeState swap %d %d, dt: %f %f", i, j, IMU_linear[i](6), IMU_linear[j](6));

        {
            std_msgs::Header tmp = Headers[i];
            Headers[i] = Headers[j];
            Headers[j] = tmp;
        }

        IMU_linear[i].swap(IMU_linear[j]);
        IMU_angular[i].swap(IMU_angular[j]);
        IMU_cov[i].swap(IMU_cov[j]);
        IMU_cov_nl[i].swap(IMU_cov_nl[j]);
        Rs[i].swap(Rs[j]);

        std::swap(imu_factors[i], imu_factors[j]);

        dt_buf[i].swap(dt_buf[j]);
        linear_acceleration_buf[i].swap(linear_acceleration_buf[j]);
        angular_velocity_buf[i].swap(angular_velocity_buf[j]);
    }
    f_manager.setRic(ric);
    printf("before shift, %d", f_manager.getFeatureCount());
    f_manager.shift(frame_count + 1 - WINDOW_SIZE);
    printf("after shift, %d", f_manager.getFeatureCount());

    delete imu_factors[WINDOW_SIZE];
    imu_factors[WINDOW_SIZE] = new IMUFactor_t{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

    IMU_linear[WINDOW_SIZE].setZero();
    IMU_angular[WINDOW_SIZE].setIdentity();
    IMU_cov[WINDOW_SIZE].setZero();
    IMU_cov_nl[WINDOW_SIZE].setZero();
    dt_buf[WINDOW_SIZE].clear();
    linear_acceleration_buf[WINDOW_SIZE].clear();
    angular_velocity_buf[WINDOW_SIZE].clear();
    frame_count = WINDOW_SIZE;

    MatrixXd tmp_A(3 + NUM_OF_CAM * 3, 3 + NUM_OF_CAM * 3);
    VectorXd tmp_b(3 + NUM_OF_CAM * 3);

    tmp_A.setZero();
    tmp_b.setZero();

    tmp_A.block<3, 3>(0, 0) = (1. / (prior_p_std * prior_p_std)) * Matrix<double, 3, 3>::Identity();
    tmp_b.segment<3>(0) = (1. / (prior_p_std * prior_p_std)) * Eigen::Vector3d::Zero();

    //for (int i = 0; i < NUM_OF_CAM; i++)
    //    if (TIC_OK[i])
    //    {
    //        tmp_A.block<3, 3>(3 + i * 3, 3 + i * 3) = (1. / (tic_std * tic_std)) * Matrix3d::Identity();
    //        tmp_b.segment<3>(3 + i * 3) = (1. / (tic_std * tic_std)) * TIC[i];
    //        tic[i] = TIC[i];
    //    }
    margin.n_Ap = tmp_A;
    margin.n_bp = tmp_b;
    margin.start_imu = 0;
    margin.number_imu = 3;
    margin.start_img = (WINDOW_SIZE + 1) * 9;
    margin.number_img = NUM_OF_CAM * 3;

    Ps[0].setZero();
    Vs[0].setZero();
    for (int i = 0; i < frame_count; i++)
    {
        int j = i + 1;
        Rs[j] = Rs[i] * IMU_angular[j];
        //Ps[j] = Rs[i] * (IMU_linear[j].segment<3>(0) + Vs[i] * dt) - g * dt * dt / 2 + Ps[i];
        //Vs[j] = Rs[j].transpose() * (Rs[i] * (IMU_linear[j].segment<3>(3) + Vs[i]) - g * dt);
        Ps[j].setZero();
        Vs[j].setZero();
    }
}

void SelfCalibrationEstimator::propagateIMU(Matrix<double, 7, 1> &imu_linear, Matrix3d &imu_angular,
                                            Matrix<double, 6, 6> &imu_cov, Matrix<double, 9, 9> &imu_cov_nl,
                                            double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity, const Vector3d &b_a, const Vector3d &b_g)
{

#if 0
    Quaterniond dq(1,
                   (angular_velocity.x() - b_g.x()) * dt / 2,
                   (angular_velocity.y() - b_g.y()) * dt / 2,
                   (angular_velocity.z() - b_g.z()) * dt / 2);
    dq.w() = 1 - dq.vec().transpose() * dq.vec();
    imu_angular = (Quaterniond(imu_angular) * dq).normalized();

    imu_linear.segment<3>(0) += imu_linear.segment<3>(3) * dt + 0.5 * imu_angular * (linear_acceleration - b_a) * dt * dt;
    imu_linear.segment<3>(3) += imu_angular * (linear_acceleration - b_a) * dt;
    imu_linear(6) += dt;
#else

    Eigen::Vector3d un_gyr = angular_velocity;
    Eigen::Vector3d un_acc = imu_angular * linear_acceleration;

    imu_linear.segment<3>(0) += imu_linear.segment<3>(3) * dt + 0.5 * un_acc * dt * dt;
    imu_linear.segment<3>(3) += un_acc * dt;
    imu_angular *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
    imu_linear(6) += dt;
#endif

    {
        Matrix<double, 6, 6> F = Matrix<double, 6, 6>::Identity();
        F.block<3, 3>(0, 3) = dt * Matrix3d::Identity();

        Matrix<double, 6, 3> G = Matrix<double, 6, 3>::Zero();
        G.block<3, 3>(0, 0) = 0.5 * dt * dt * Matrix3d::Identity();
        G.block<3, 3>(3, 0) = dt * Matrix3d::Identity();

        imu_cov = F * imu_cov * F.transpose() + G * acc_cov * G.transpose();
    }

    {
        Matrix<double, 9, 9> F = Matrix<double, 9, 9>::Zero();
        F.block<3, 3>(0, 3) = Matrix3d::Identity();
        F.block<3, 3>(3, 6) = -imu_angular * Utility::skewSymmetric(linear_acceleration);
        F.block<3, 3>(6, 6) = -Utility::skewSymmetric(angular_velocity);

        Matrix<double, 6, 6> Q = Matrix<double, 6, 6>::Zero();
        Q.block<3, 3>(0, 0) = acc_cov;
        Q.block<3, 3>(3, 3) = gyr_cov;

        Matrix<double, 9, 6> G = Matrix<double, 9, 6>::Zero();
        G.block<3, 3>(3, 0) = -imu_angular;
        G.block<3, 3>(6, 3) = -Matrix3d::Identity();

        imu_cov_nl = (Matrix<double, 9, 9>::Identity() + dt * F) * imu_cov_nl * (Matrix<double, 9, 9>::Identity() + dt * F).transpose() +
                     (dt * G) * Q * (dt * G).transpose();
    }
}

void SelfCalibrationEstimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!imu_factors[frame_count])
    {
        imu_factors[frame_count] = new IMUFactor_t{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }

    if (frame_count == 0)
    {
        //Bgs[0] = angular_velocity;
        //cout << "here" << endl;
        //cout << Bgs[0] << endl;
    }
    else
    {
        if (true)
        {
            g_sum += linear_acceleration;
            g_cnt++;
            ROS_INFO("average gravity %f", (g_sum / g_cnt).norm());
        }

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        propagateIMU(IMU_linear[frame_count], IMU_angular[frame_count],
                     IMU_cov[frame_count], IMU_cov_nl[frame_count],
                     dt, linear_acceleration, angular_velocity, Bas[frame_count - 1], Bgs[frame_count - 1]);

        imu_factors[frame_count]->push_back(dt, linear_acceleration, angular_velocity);

#if 1
        int j = frame_count;
        Eigen::Vector3d tmp_P = Ps[j];
        Eigen::Quaterniond tmp_R{Rs[j]};
        Eigen::Vector3d tmp_V = Vs[j];
        Eigen::Vector3d tmp_Ba = Bas[j];
        Eigen::Vector3d tmp_Bg = Bgs[j];

        Eigen::Quaterniond tmp_Q;

        Eigen::Vector3d tmp_acc_0 = acc_0 - Rs[j].transpose() * g;
        Eigen::Vector3d tmp_acc_1 = linear_acceleration - Rs[j].transpose() * g;

        imu_factors[frame_count]->pre_integration.propagate_implementation(
            tmp_P, tmp_R, tmp_V, tmp_Ba, tmp_Bg,
            tmp_acc_0, gyr_0, tmp_acc_1, angular_velocity,
            Ps[j], tmp_Q, Vs[j], Bas[j], Bgs[j]);
        Rs[j] = tmp_Q.toRotationMatrix();
#else
        //TODO: unified integration
        int j = frame_count;
        Vector3d un_acc = Rs[j] * (linear_acceleration - Bas[j]) - Vector3d(0.0, 0.0, 9.81);
        Vector3d un_gyr = (angular_velocity - Bgs[j]);
        Ps[j] += Vs[j] * dt + 0.5 * un_acc * dt * dt;
        Vs[j] += un_acc * dt;
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
#endif
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

#define Y_COS 2
#define Z_COS 2 * 2

Vector3d getPosition(double t)
{
    static int const MAX_BOX = 10;
    static int const MAX_TIME = 10;
    double x, y, z;
    if (t < MAX_TIME)
    {
        x = MAX_BOX / 2.0 + MAX_BOX / 2.0 * cos(t / MAX_TIME * M_PI);
        y = MAX_BOX / 2.0 + MAX_BOX / 2.0 * cos(t / MAX_TIME * M_PI * Y_COS);
        z = MAX_BOX / 2.0 + MAX_BOX / 2.0 * cos(t / MAX_TIME * M_PI * Z_COS);
    }
    else if (t >= MAX_TIME && t < 2 * MAX_TIME)
    {
        x = MAX_BOX / 2.0 - MAX_BOX / 2.0;
        y = MAX_BOX / 2.0 + MAX_BOX / 2.0;
        z = MAX_BOX / 2.0 + MAX_BOX / 2.0;
    }
    else
    {
        double tt = t - 2 * MAX_TIME;
        x = -MAX_BOX / 2.0 + MAX_BOX / 2.0 * cos(tt / MAX_TIME * M_PI);
        y = MAX_BOX / 2.0 + MAX_BOX / 2.0 * cos(tt / MAX_TIME * M_PI * Y_COS);
        z = MAX_BOX / 2.0 + MAX_BOX / 2.0 * cos(tt / MAX_TIME * M_PI * Z_COS);
    }

    return Vector3d(x, y, z);
}

Matrix3d getRotation(double t)
{
    static int const MAX_TIME = 10;
    return (AngleAxisd(30.0 / 180 * M_PI * sin(t / MAX_TIME * M_PI * 2), Vector3d::UnitX()) * AngleAxisd(40.0 / 180 * M_PI * sin(t / MAX_TIME * M_PI * 2), Vector3d::UnitY()) * AngleAxisd(0, Vector3d::UnitZ())).toRotationMatrix();
}

Vector3d getVelocity(double t)
{
    static int const MAX_BOX = 10;
    static int const MAX_TIME = 10;
    double dx, dy, dz;
    if (t < MAX_TIME)
    {
        dx = MAX_BOX / 2.0 * -sin(t / MAX_TIME * M_PI) * (1.0 / MAX_TIME * M_PI);
        dy = MAX_BOX / 2.0 * -sin(t / MAX_TIME * M_PI * Y_COS) * (1.0 / MAX_TIME * M_PI * Y_COS);
        dz = MAX_BOX / 2.0 * -sin(t / MAX_TIME * M_PI * Z_COS) * (1.0 / MAX_TIME * M_PI * Z_COS);
    }
    else if (t >= MAX_TIME && t < 2 * MAX_TIME)
    {
        dx = 0.0;
        dy = 0.0;
        dz = 0.0;
    }
    else
    {
        double tt = t - 2 * MAX_TIME;
        dx = MAX_BOX / 2.0 * -sin(tt / MAX_TIME * M_PI) * (1.0 / MAX_TIME * M_PI);
        dy = MAX_BOX / 2.0 * -sin(tt / MAX_TIME * M_PI * Y_COS) * (1.0 / MAX_TIME * M_PI * Y_COS);
        dz = MAX_BOX / 2.0 * -sin(tt / MAX_TIME * M_PI * Z_COS) * (1.0 / MAX_TIME * M_PI * Z_COS);
    }

    return Vector3d(dx, dy, dz);
}

void SelfCalibrationEstimator::processImage(const map<int, vector<pair<int, Vector3d>>> &image, const std_msgs::Header &header,
                                            const map<int, Vector3d> &debug_image)
{
    ROS_INFO("Adding feature points %lu", image.size());
    marginalization_flag = !f_manager.addFeatureCheckParallax(frame_count, image, debug_image);
    if (SHOW_HTML)
        f_manager.htmlShow();
    ROS_DEBUG("marginalization_flag %d", int(marginalization_flag));
    ROS_INFO("this frame is-------------------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_INFO("Solving %d", frame_count);
    ROS_INFO("number of feature: %d", f_manager.getFeatureCount());

    Headers[frame_count] = header;

    if (solver_flag == CALIBRATION)
    {
        double cali_cov = 1.;
        for (int i = 0; i < NUM_OF_CAM; i++)
            cali_cov = min(cali_cov, solveCalibration(i));
        is_valid = true;
        frame_count++;
        if (frame_count >= WINDOW_SIZE && cali_cov > CALIB_THRESHOLD_RIC && f_manager.getFeatureCount() >= 10)
        {
            puts("switch");
            for (int i = 0; i < NUM_OF_CAM; i++)
                ric[i] = RIC[i];
            changeState();
        }
    }
    else
    {
        TicToc t_solve;
        solveOdometry();
        ROS_INFO("solver costs: %fms", t_solve.toc());

        TicToc t_margin;
        setPrior();
        marginalize();
        ROS_INFO("marginalization costs: %fms", t_margin.toc());

        is_key = !marginalization_flag;

        ROS_INFO("number of feature: %d", f_manager.getFeatureCount());
        ROS_INFO("number of outlier: %d", sum_of_outlier);
        ROS_ASSERT(sum_of_outlier == 0);
        ROS_INFO("number of diverge: %d", sum_of_diverge);
        ROS_INFO("number of invalid: %d", sum_of_invalid);
        ROS_INFO("number of back: %d", sum_of_back);
        ROS_INFO("number of front: %d", sum_of_front);
    }
}

double SelfCalibrationEstimator::solveCalibration(int camera_id)
{
    if (frame_count == 0)
    {
        return 0;
    }

    vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count, camera_id);
    Rc[camera_id][frame_count] = m_estimator.solveRelativeR(corres);

    //ROS_DEBUG_STREAM("motion_estimator:\n" << Utility::R2ypr(Rc[frame_count]).transpose());
    //ROS_DEBUG_STREAM("motion_estimator:\n" << Rc[frame_count]);

    Rc_g[camera_id][frame_count] = ric[camera_id].inverse() * IMU_angular[frame_count] * ric[camera_id];
    //ROS_DEBUG_STREAM("ground_true:\n" << Utility::R2ypr(Rc_g[frame_count]).transpose());
    //ROS_DEBUG_STREAM("ground_true:\n" << Rc_g[frame_count]);

    Eigen::MatrixXd A(frame_count * 4, 4);
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++)
    {
        Quaterniond r1(Rc[camera_id][i]);
        Quaterniond r2(Rc_g[camera_id][i]);

        double angular_distance = 180 / M_PI * r1.angularDistance(r2);
        ROS_DEBUG("%d %f", i, angular_distance);

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
        ++sum_ok;
        Matrix4d L, R;

        double w = Quaterniond(Rc[camera_id][i]).w();
        Vector3d q = Quaterniond(Rc[camera_id][i]).vec();
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q);
        L.block<3, 1>(0, 3) = q;
        L.block<1, 3>(3, 0) = -q.transpose();
        L(3, 3) = w;

        Quaterniond R_ij(IMU_angular[i]);
        w = R_ij.w();
        q = R_ij.vec();
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q);
        R.block<3, 1>(0, 3) = q;
        R.block<1, 3>(3, 0) = -q.transpose();
        R(3, 3) = w;

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
    Matrix<double, 4, 1> x = svd.matrixV().col(3);
    Quaterniond estimated_R(x);

    ric[camera_id] = estimated_R.toRotationMatrix().inverse();
    ROS_INFO("calibration %d", camera_id);
    ROS_INFO_STREAM("\n" << ric[camera_id]);
    ric_cov[camera_id] = svd.singularValues().tail<3>();
    ROS_INFO_STREAM(svd.singularValues().transpose());

    return ric_cov[camera_id](1);
}

void SelfCalibrationEstimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
    {
        return;
    }
    if (solver_flag == LINEAR)
    {
        int n_state = (frame_count + 1) * 9 + NUM_OF_CAM * 3 + f_manager.getFeatureCount();

        f_manager.triangulate(Ps, tic, ric);

        MatrixXd A(n_state, n_state);
        VectorXd b(n_state);
        for (int i = 0; i < NUM_OF_ITER; i++)
        {
            A.setZero();
            b.setZero();
            margin.setPrior(A, b);
            solveOdometryLinear(A, b);
        }

        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            last_ric[i] = Utility::R2ypr(ric[i]);
            last_tic[i] = tic[i];
        }

        if (inv_cnt++ % 2 == 0)
        {
            TicToc t_inverse;
            ROS_INFO("inverse");
            MatrixXd A_inv = A.inverse();
            VectorXd cov_diag = A_inv.diagonal();
            ROS_INFO("solving inverse costs: %fms", t_inverse.toc());
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                tic_cov[i] = cov_diag.segment<3>((frame_count + 1) * 9 + i * 3);
                tic_cov[i](0) = sqrt(tic_cov[i](0));
                tic_cov[i](1) = sqrt(tic_cov[i](1));
                tic_cov[i](2) = sqrt(tic_cov[i](2));
            }
        }

        double max_cov = 0.0;
        for (int i = 0; i < NUM_OF_CAM; i++)
            if (tic_cov[i].maxCoeff() > max_cov)
                max_cov = tic_cov[i].maxCoeff();

        //if (fabs(9.8 - g.norm()) < 0.05 && f_manager.getFeatureCount() >= 10)
        if (true)
        {
            MatrixXd tmp_A(15 + NUM_OF_CAM * 6, 15 + NUM_OF_CAM * 6);
            VectorXd tmp_b(15 + NUM_OF_CAM * 6);

            tmp_A.setZero();
            tmp_b.setZero();

            tmp_A.block<3, 3>(0, 0) = (1. / (prior_p_std * prior_p_std)) * Matrix<double, 3, 3>::Identity();

            tmp_A.block<3, 3>(6, 6) = (1. / (prior_q_std * prior_q_std)) * Matrix<double, 3, 3>::Identity();

            tmp_A.block<3, 3>(9, 9) = (1. / (0.50 * 0.50)) * Matrix<double, 3, 3>::Identity();

            tmp_A.block<3, 3>(12, 12) = (1. / (0.50 * 0.50)) * Matrix<double, 3, 3>::Identity();

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                tmp_A.block<3, 3>(15 + i * 6 + 0, 15 + i * 6 + 0) = (1. / (tic_std * tic_std)) * Matrix3d::Identity();
                tmp_A.block<3, 3>(15 + i * 6 + 3, 15 + i * 6 + 3) = (1. / (ric_std * ric_std)) * Matrix3d::Identity();
            }

            margin.n_Ap = tmp_A;
            margin.n_bp = tmp_b;
            margin.start_imu = 0;
            margin.number_imu = 15;
            margin.start_img = (WINDOW_SIZE + 1) * 15;
            margin.number_img = NUM_OF_CAM * 6;

            if (true) // skip linear calibration
            {
                for (int i = 0; i < NUM_OF_CAM; i++)
                {
                    if (TIC_OK[i])
                    {
                        tic[i] = TIC[i];
                    }
                    if (RIC_OK[i])
                    {
                        ric[i] = RIC[i];
                    }
                }

#ifdef GT
                g = G;
                Ps[0] = getPosition(Headers[0].stamp.toSec());
                Vs[0] = getVelocity(Headers[0].stamp.toSec());
                Rs[0] = getRotation(Headers[0].stamp.toSec());
#else
                ROS_INFO_STREAM("gravity init " << g.transpose());
                Rs[0] = Utility::g2R(g);
                ROS_ASSERT(Utility::R2ypr(Rs[0]).x() <= 1e-6);
                g = Rs[0] * g.normalized() * G.norm();
                ROS_INFO_STREAM("reset gravity " << g.transpose());
                ROS_INFO_STREAM("origin gravity " << (Rs[0].transpose() * g).transpose());
                ROS_INFO_STREAM("R0:\n" << Rs[0]);

#endif

//old2new();
//init_factor = new PoseFactor(Ps[0], Quaterniond(Rs[0]));
//problem.AddResidualBlock(init_factor, NULL, para_Pose[0]);
#if 0
                double **para = new double *[1];
                para[0] = para_Pose[0];
                f->checkJacobian(para);

                ROS_BREAK();
#endif

//for (int i = 0; i < frame_count; i++)
//{
//    Vs[i] = Rs[i] * Vs[i];
//    cout << i << " " << Vs[i].transpose() << endl;
//}
#ifdef GT
                printf("%d, %f, %f %f %f, %f %f %f\n", 0, Headers[0].stamp.toSec(), Ps[0].x() - Ps[0].x(), Ps[0].y() - Ps[0].y(), Ps[0].z() - Ps[0].z(),
                       Vs[0].x(), Vs[0].y(), Vs[0].z());
                std::cout << Rs[0] << std::endl;

                for (int i = 0; i < frame_count; i++)
                {
                    int j = i + 1;
                    double dt = IMU_linear[j](6);
                    Rs[j] = Rs[i] * IMU_angular[j];
                    Ps[j] = Rs[i] * IMU_linear[j].segment<3>(0) + Vs[i] * dt - 0.5 * g * dt * dt + Ps[i];
                    Vs[j] = Rs[i] * IMU_linear[j].segment<3>(3) + Vs[i] - g * dt;

                    printf("%d, %f, %f %f %f, %f %f %f\n", j, Headers[j].stamp.toSec(), Ps[j].x() - Ps[0].x(), Ps[j].y() - Ps[0].y(), Ps[j].z() - Ps[0].z(),
                           Vs[j].x(), Vs[j].y(), Vs[j].z());
                    std::cout << Rs[j] << std::endl;
                }

                for (int i = 0; i <= frame_count; i++)
                {
                    double t = Headers[i].stamp.toSec();
                    printf("%f\n", t);

                    Ps[i] = getPosition(t);
                    Vs[i] = getVelocity(t);
                    Rs[i] = getRotation(t);

                    printf("%d, %f, %f %f %f, %f %f %f\n", i, Headers[i].stamp.toSec(), Ps[i].x() - Ps[0].x(), Ps[i].y() - Ps[0].y(), Ps[i].z() - Ps[0].z(),
                           Vs[i].x(), Vs[i].y(), Vs[i].z());
                    std::cout << Rs[i] << std::endl;
                }
#else

                for (int i = 0; i < frame_count; i++)
                {
                    int j = i + 1;
                    double dt = IMU_linear[j](6);
                    Rs[j] = Rs[i] * IMU_angular[j];
                    Ps[j] = Rs[i] * IMU_linear[j].segment<3>(0) + Vs[i] * dt - 0.5 * g * dt * dt + Ps[i];
                    Vs[j] = Rs[i] * IMU_linear[j].segment<3>(3) + Vs[i] - g * dt;
                }
#endif
                //for (int i = 0; i < frame_count; i++)
                //{
                //    Vs[i] = Rs[i] * Vs[i];
                //    cout << i << " " << Vs[i].transpose() << endl;
                //}

                VectorXd dep = f_manager.getDepthVector();
                for (int i = 0; i < dep.size(); i++)
                    dep[i] = -1;
                f_manager.setDepth(dep);
                f_manager.triangulate(Ps, tic, ric);
            }
            //ROS_BREAK();

            for (int i = 0; i <= WINDOW_SIZE; i++)
                use_cov[i] = 1;

            //solveOdometryNonlinear(true);
            solve_ceres();
            //ROS_BREAK();

            solver_flag = NON_LINEAR;
        }
    }
    else if (solver_flag == NON_LINEAR)
    {
        TicToc t_prepare;
        init_P = Ps[WINDOW_SIZE];
        init_V = Vs[WINDOW_SIZE];
        init_R = Rs[WINDOW_SIZE];

        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        outlier_info.clear();
        for (auto outlier : f_manager.outlier_info)
        {
            std::vector<long long> header_time;
            printf("id: %d ", outlier.first);
            for (auto j : outlier.second)
            {
                printf("%d ", j);
                header_time.push_back(Headers[j].stamp.toNSec());
            }
            puts("");
            outlier_info.emplace_back(outlier.first, header_time);
        }
        ROS_INFO("triangulation costs %f", t_tri.toc());

        TicToc t_whole_nonlinear;
        //solveOdometryNonlinear(false);

        solve_ceres();
        //ROS_BREAK();
        ROS_DEBUG("non_linear cost: %f", t_whole_nonlinear.toc());
    }

    //ROS_INFO("rejecting outlier");
    //rejectOutlier();
    TicToc t_post_solver;

    ROS_DEBUG("post solver cost1: %f", t_post_solver.toc());
    t_post_solver.tic();

    ROS_DEBUG("post solver cost2: %f", t_post_solver.toc());
    t_post_solver.tic();

    // prepare output of VINS
    key_poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
        key_poses.push_back(Ps[i]);

    point_cloud.clear();
    for (auto it_per_id : f_manager.traversal())
    {
        ROS_DEBUG("dep: %f", it_per_id->estimated_depth);
        int imu_i = it_per_id->start_frame;
        int camera_i = it_per_id->feature_per_frame[0].feature_per_camera[0].camera_id;
        Vector3d pts_i = it_per_id->feature_per_frame[0].feature_per_camera[0].point * it_per_id->estimated_depth;
        point_cloud.push_back(Rs[imu_i] * (ric[camera_i] * pts_i + tic[camera_i]) + Ps[imu_i]);
    }

    is_valid = (solver_flag == LINEAR || solver_flag == CALIBRATION);
    sum_of_invalid += !is_valid;

    ROS_DEBUG("post solver cost3: %f", t_post_solver.toc());
}

void SelfCalibrationEstimator::setPrior()
{
    ROS_DEBUG("remove outlier's info from prior: minus");
    int n_state = bp[marginalization_flag].rows();

    f_manager.tagMarginalizedPoints(solver_flag == NON_LINEAR, marginalization_flag);
    return;

    margin_map.clear();

    int feature_index = n_state - f_manager.getFeatureCount() - 1;
    for (auto it_per_id : f_manager.traversal())
    {
        ++feature_index;
        int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
        int camera_i = it_per_id->feature_per_frame[0].feature_per_camera[0].camera_id, camera_j;

        // output marginalized point as map
        if (it_per_id->is_margin && !marginalization_flag)
        {
            int imu_i = it_per_id->start_frame;
            int camera_i = it_per_id->feature_per_frame[0].feature_per_camera[0].camera_id;
            Vector3d pts_i = it_per_id->feature_per_frame[0].feature_per_camera[0].point * it_per_id->estimated_depth;
            margin_map.push_back(Rs[imu_i] * (ric[camera_i] * pts_i + tic[camera_i]) + Ps[imu_i]);
        }

        for (auto &it_per_frame : it_per_id->feature_per_frame)
        {
            imu_j++;
            for (auto &it_per_camera : it_per_frame.feature_per_camera)
            {
                camera_j = it_per_camera.camera_id;
                if (imu_i == imu_j && camera_i == camera_j)
                {
#ifdef DEPTH_PRIOR
                    if (solver_flag == LINEAR)
                    {
                        double var = 1 / (std::max(5.0 - 1.0, 10.0 - 5.0) / 3);
                        Ap[marginalization_flag](feature_index, feature_index) += var * var;
                        bp[marginalization_flag](feature_index) += var * var * 5.0;
                    }
                    if (solver_flag == NON_LINEAR)
                    {
                        double var = 1 / (std::max(1 / 1.0 - 1 / 5.0, 1 / 5.0 - 1 / 10.0) / 3);
                        Ap[marginalization_flag](feature_index, feature_index) += var * var;
                    }
#endif
                    continue;
                }
                if (it_per_id->is_margin) // || (marginalization_flag && imu_j == frame_count - 1))
                {
                    if (solver_flag == LINEAR)
                    {
                        int loop_p[] = {imu_i * 9, imu_j * 9, (frame_count + 1) * 9, feature_index};
                        int loop_p_copy[] = {0, 3, 6, 6 + NUM_OF_CAM * 3};
                        int loop_s[] = {3, 3, NUM_OF_CAM * 3, 1};
                        for (int ii = 0; ii < 4; ii++)
                            for (int jj = 0; jj < 4; jj++)
                                Ap[marginalization_flag].block(loop_p[ii], loop_p[jj], loop_s[ii], loop_s[jj]) += it_per_camera.A.block(loop_p_copy[ii], loop_p_copy[jj], loop_s[ii], loop_s[jj]);
                    }
                    else if (solver_flag == NON_LINEAR)
                    {
                        int n_cam = camera_i == camera_j ? 1 : 2;
                        int n_v_state = 6 + 6 + n_cam * 6 + 1;

                        vector<int> loop_p({imu_i * 15 + 0, imu_i * 15 + 6, imu_j * 15 + 0, imu_j * 15 + 6});
                        vector<int> loop_p_copy({0, 3, 6, 9});
                        vector<int> loop_s({3, 3, 3, 3});

                        loop_p.push_back((frame_count + 1) * 15 + camera_i * 6);
                        loop_p_copy.push_back(12);
                        loop_s.push_back(6);

                        if (camera_i != camera_j)
                        {
                            loop_p.push_back((frame_count + 1) * 15 + camera_j * 6);
                            loop_p_copy.push_back(18);
                            loop_s.push_back(6);
                        }
                        loop_p.push_back(feature_index);
                        loop_p_copy.push_back(n_v_state - 1);
                        loop_s.push_back(1);

                        for (unsigned int ii = 0; ii < loop_p.size(); ii++)
                        {
                            bp[marginalization_flag].segment(loop_p[ii], loop_s[ii]) += it_per_camera.b.segment(loop_p_copy[ii], loop_s[ii]);
                            for (unsigned int jj = 0; jj < loop_p.size(); jj++)
                                Ap[marginalization_flag].block(loop_p[ii], loop_p[jj], loop_s[ii], loop_s[jj]) += it_per_camera.A.block(loop_p_copy[ii], loop_p_copy[jj], loop_s[ii], loop_s[jj]);
                        }
                    }
                }
            }
        }
    }
}

void SelfCalibrationEstimator::solveOdometryLinear(MatrixXd &A, VectorXd &b)
{
    TicToc t_whole;
    int n_state = (frame_count + 1) * 9 + NUM_OF_CAM * 3 + f_manager.getFeatureCount();

    ROS_DEBUG("setting prior");
    TicToc t_p;

    Ap[0] = Ap[1] = A;
    bp[0] = bp[1] = b;
    ROS_DEBUG("feature_count: %d", f_manager.getFeatureCount());

    ROS_DEBUG("setting prior costs: %fms", t_p.toc());

    ROS_DEBUG("setting imu");
    TicToc t_i;
    for (int i = 0; i < frame_count; i++)
    {
        int j = i + 1;
        if (use_cov[j] != 1)
            continue;
        MatrixXd tmp_A(9, n_state);
        tmp_A.setZero();
        VectorXd tmp_b(9);
        tmp_b.setZero();

        Matrix3d r_i_inv = Rs[i].transpose();

        double dt = IMU_linear[i + 1](6);
        ROS_DEBUG("%d dt: %f", i, dt);
        tmp_A.block<3, 3>(0, i * 9) = -r_i_inv;
        tmp_A.block<3, 3>(0, i * 9 + 3) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, i * 9 + 6) = dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, (i + 1) * 9) = r_i_inv;
        tmp_b.block<3, 1>(0, 0) = IMU_linear[i + 1].segment<3>(0);

        tmp_A.block<3, 3>(3, (i + 1) * 9 + 3) = IMU_angular[i + 1];
        tmp_A.block<3, 3>(3, i * 9 + 3) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, i * 9 + 6) = dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = IMU_linear[i + 1].segment<3>(3);

        tmp_A.block<3, 3>(6, (i + 1) * 9 + 6) = IMU_angular[i + 1];
        tmp_A.block<3, 3>(6, i * 9 + 6) = -Matrix3d::Identity();
        tmp_b.block<3, 1>(6, 0) = Vector3d::Zero();

        Matrix<double, 9, 9> cov = Matrix<double, 9, 9>::Zero();
        cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        cov.block<3, 3>(6, 6) = gra_cov;

        MatrixXd cov_inv = cov.inverse();

        MatrixXd r_A = tmp_A.block<9, 18>(0, i * 9).transpose() * cov_inv * tmp_A.block<9, 18>(0, i * 9);
        VectorXd r_b = tmp_A.block<9, 18>(0, i * 9).transpose() * cov_inv * tmp_b;

        A.block<18, 18>(i * 9, i * 9) += r_A;
        b.segment<18>(i * 9) += r_b;
        if (!marginalization_flag && i == 0)
        {
            Ap[0].block<18, 18>(i * 9, i * 9) += r_A;
            bp[0].segment<18>(i * 9) += r_b;
        }
        if (marginalization_flag && (j == frame_count - 1 || j == frame_count))
        {
            Ap[1].block<18, 18>(i * 9, i * 9) += r_A;
            bp[1].segment<18>(i * 9) += r_b;
        }
    }
    ROS_DEBUG("setting imu costs: %fms", t_i.toc());

    ROS_DEBUG("setting vision");
    TicToc t_v;

    int feature_index = b.rows() - f_manager.getFeatureCount() - 1;
    ROS_DEBUG("all state %d", n_state);
    for (auto it_per_id : f_manager.traversal())
    {
        ++feature_index;

        int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
        int camera_i = it_per_id->feature_per_frame[0].feature_per_camera[0].camera_id, camera_j;
        Vector3d p_i = Ps[imu_i], p_j;

        Vector3d pts_i = it_per_id->feature_per_frame[0].feature_per_camera[0].point, pts_j;
        double dep_i = it_per_id->estimated_depth < 0.1 ? INIT_DEPTH : it_per_id->estimated_depth, dep_j;

        Vector3d pts_camera_i = pts_i * dep_i, pts_camera_j;
        Vector3d pts_imu_i = ric[camera_i] * pts_camera_i + tic[camera_i];
        Vector3d pts_world = Rs[imu_i] * pts_imu_i + p_i;

        Matrix3d r_i = Rs[imu_i];
        Matrix3d ric_i = ric[camera_i];

        for (auto &it_per_frame : it_per_id->feature_per_frame)
        {
            imu_j++;
            //printf("%d ", it_per_id->feature_id);
            for (auto &it_per_camera : it_per_frame.feature_per_camera)
            {
                camera_j = it_per_camera.camera_id;
                if (imu_i == imu_j && camera_i == camera_j)
                {
#ifdef DEPTH_PRIOR
                    double var = 1 / (std::max(5.0 - 1.0, 10.0 - 5.0) / 3);
                    double w = 10.0;
                    A(feature_index, feature_index) += w * var * var;
                    b(feature_index) += w * var * var * 5.0;
#endif
                    continue;
                }
                p_j = Ps[imu_j];
                pts_j = it_per_camera.point;

                Matrix3d r_j_inv = Rs[imu_j].transpose();
                Matrix3d r_j_i = r_j_inv * r_i;
                Matrix3d ric_j_inv = ric[camera_j].transpose();

                Vector3d pts_imu_j = r_j_inv * (pts_world - p_j);
                Vector3d pts_camera_j = ric_j_inv * (pts_imu_j - tic[camera_j]);
                dep_j = pts_camera_j(2);

                VectorXd tmp_b = Vector2d(pts_camera_j(0) / dep_j - pts_j(0),
                                          pts_camera_j(1) / dep_j - pts_j(1));
                //printf("%f %d:%d %d (%f %f) -> (%f %f) %f\n", dep_i, it_per_id->feature_id, camera_i, camera_j, pts_camera_j(0) / dep_j, pts_camera_j(1) / dep_j, pts_j(0), pts_j(1), tmp_b.norm());
                //printf("(%f %f) %f, %f, ", tmp_b(0), tmp_b(1), dep_i, it_per_id->estimated_depth);

                Matrix<double, 2, 3> reduce;
                reduce << 1, 0, -pts_j(0),
                    0, 1, -pts_j(1);

                MatrixXd tmp_A(3, 3 + 3 + NUM_OF_CAM * 3 + 1);
                tmp_A.block<3, NUM_OF_CAM * 3>(0, 6).setZero();

                tmp_A.block<3, 3>(0, 0) = r_j_inv;
                tmp_A.block<3, 3>(0, 3) = -r_j_inv;
                tmp_A.block<3, 3>(0, 6 + camera_i * 3) += r_j_i;
                tmp_A.block<3, 3>(0, 6 + camera_j * 3) += -Matrix3d::Identity();
                tmp_A.block<3, 1>(0, 6 + NUM_OF_CAM * 3) = r_j_i * ric_i * pts_i;

                tmp_A = reduce * ric_j_inv * tmp_A;

                int loop_p[] = {imu_i * 9, imu_j * 9, (frame_count + 1) * 9, feature_index};
                int loop_p_copy[] = {0, 3, 6, 6 + NUM_OF_CAM * 3};
                int loop_s[] = {3, 3, NUM_OF_CAM * 3, 1};

                //huber norm
                double err = tmp_b.norm();
                double w = err > FEATURE_THRESHOLD ? FEATURE_THRESHOLD / err : 1.0;
                w = 1.0;

                // the depth of pts_camera_i is based previous estimation
                Matrix2d pts_cov_inv = 1. / dep_j / dep_j * pts_cov.inverse();
                //printf("%f", pts_cov_inv);

                it_per_camera.A = w * tmp_A.transpose() * pts_cov_inv * tmp_A;

                for (int ii = 0; ii < 4; ii++)
                    for (int jj = 0; jj < 4; jj++)
                        A.block(loop_p[ii], loop_p[jj], loop_s[ii], loop_s[jj]) += it_per_camera.A.block(loop_p_copy[ii], loop_p_copy[jj], loop_s[ii], loop_s[jj]);
            }
            //puts("");
        }
    }
    ROS_DEBUG("setting vision costs: %fms", t_v.toc());
    ROS_DEBUG("feature_index: %d, n_state: %d", feature_index, n_state);
    VectorXd x = A.ldlt().solve(b);

    //ColPivHouseholderQR<MatrixXd> solver(A);
    //if (solver.rank() != A.rows())
    //{
    //    ROS_INFO("rank fails");
    //    return;
    //}
    //VectorXd x = solver.solve(b);
    //ROS_DEBUG("setting vision costs: %fms", t_whole.toc());
    //if (b.isApprox(A * x))
    //    ROS_INFO("linear solver finds solution");
    //else
    //{
    //    ROS_INFO("linear solver fails");
    //    return;
    //}

    g.setZero();
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = x.segment<3>(i * 9);
        Vs[i] = x.segment<3>(i * 9 + 3);
        g += Rs[i] * x.segment<3>(i * 9 + 6);
        ROS_DEBUG_STREAM((Rs[i] * x.segment<3>(i * 9 + 6)).transpose());
    }
    g /= frame_count + 1;

    for (int i = 0; i < NUM_OF_CAM; i++)
        tic[i] = x.segment<3>((frame_count + 1) * 9 + i * 3);

    f_manager.setDepth(x.tail(f_manager.getFeatureCount()));
}

void SelfCalibrationEstimator::old2new()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    ROS_INFO("old2new");
    //for (int i = WINDOW_SIZE - 3; i <= WINDOW_SIZE; i++)
    //    ROS_INFO_STREAM(i << ": " << Ps[i].transpose() << " " << Vs[i].transpose());
}

void SelfCalibrationEstimator::new2old()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    ROS_INFO("origin R %f %f %f", origin_R0.x(), origin_R0.y(), origin_R0.z());

    Vector3d origin_P0 = Ps[0];
    ROS_INFO("origin P %f %f %f", origin_P0.x(), origin_P0.y(), origin_P0.z());

    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5]).toRotationMatrix());

    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
#if 1
        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) +

                origin_P0;
        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);
#else
        Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).toRotationMatrix();
        Ps[i] = Vector3d(para_Pose[i][0],
                         para_Pose[i][1],
                         para_Pose[i][2]);
        Vs[i] = Vector3d(para_SpeedBias[i][0],
                         para_SpeedBias[i][1],
                         para_SpeedBias[i][2]);
#endif

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }
    Vector3d cur_P0 = Ps[0];
    ROS_INFO("current P %f %f %f", cur_P0.x(), cur_P0.y(), cur_P0.z());

    Vector3d cur_R0 = Utility::R2ypr(Rs[0]);
    ROS_INFO("current R %f %f %f", cur_R0.x(), cur_R0.y(), cur_R0.z());

#if 1
    ROS_ASSERT((origin_P0 - cur_P0).norm() < 1e-6);
    ROS_ASSERT((origin_R0.x() - cur_R0.x()) < 1e-6);
#endif

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    ROS_INFO("new2new");
    //for (int i = WINDOW_SIZE - 3; i <= WINDOW_SIZE; i++)
    //    ROS_INFO_STREAM(i << ": " << Ps[i].transpose() << " " << Vs[i].transpose());
}

void SelfCalibrationEstimator::solve_ceres()
{
    TicToc t_whole, t_prepare;
    old2new();

    IMUFactor_t::sum_t = 0.0;
    ProjectionFactor::sum_t = 0.0;

    double sum_error_ceres = 0;

    TicToc t_cost1;
    {
        double m_sum = 0;
        if (last_marginalization_factor)
        {
            problem.AddResidualBlock(last_marginalization_factor, NULL,
                                     last_marginalization_parameter_blocks);
#if 1
            double **para = new double *[last_marginalization_parameter_blocks.size()];
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                para[i] = last_marginalization_parameter_blocks[i];
            double *res = new double[last_marginalization_factor->num_residuals()];

            last_marginalization_factor->Evaluate(para, res, NULL);
            for (int i = 0; i < last_marginalization_factor->num_residuals(); i++)
            {
                m_sum += res[i] * res[i];
            }
#endif
        }
        ROS_INFO("marginalization error: %f", m_sum);
        sum_error_ceres += m_sum;
    }

    {
        double i_sum = 0.0;
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            int j = i + 1;

            problem.AddResidualBlock(imu_factors[j], NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);

//printf("%f %f %f %f %f %f %f\n", para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6]);

//imu_factors[j]->checkTransition();
//imu_factors[j]->checkBias();

#if 0
            double **para = new double *[4];
            para[0] = para_Pose[i];
            para[1] = para_SpeedBias[i];
            para[2] = para_Pose[j];
            para[3] = para_SpeedBias[j];
            //double *tmp_r = new double[15];
            //imu_factors[j]->Evaluate(para, tmp_r, NULL);
            //double tmp_sum = 0.0;
            //for (int j = 0; j < 15; j++)
            //{
            //    tmp_sum += tmp_r[j] * tmp_r[j];
            //    printf("%f ", tmp_r[j] * tmp_r[j]);
            //}
            //puts("");
            //i_sum += tmp_sum;
            //printf("dt: %f %f\n", IMU_linear[j](6), tmp_sum);
            //cout << IMU_linear[j].transpose() << endl;
            //cout << imu_factors[j]->delta_p.transpose() << " " << imu_factors[j]->delta_v.transpose() << endl;
            imu_factors[j]->checkJacobian(para);
#endif
        }
        //puts("");
        ROS_INFO("imu error: %f", i_sum);
        sum_error_ceres += i_sum;
    }

    {
        int f_m_cnt = 0;
        double f_sum = 0.0;
        double r_f_sum = 0.0;
        int feature_index = -1;
        for (auto it_per_id : f_manager.traversal())
        {
            ++feature_index;

            int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
            int camera_i = it_per_id->feature_per_frame[0].feature_per_camera[0].camera_id;

            Vector3d pts_i = it_per_id->feature_per_frame[0].feature_per_camera[0].point;

            for (auto &it_per_frame : it_per_id->feature_per_frame)
            {
                imu_j++;
                for (auto &it_per_camera : it_per_frame.feature_per_camera)
                {
                    int camera_j = it_per_camera.camera_id;
                    if (imu_i == imu_j && camera_i == camera_j)
                    {
                        continue;
                    }
                    Vector3d pts_j = it_per_camera.point;
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);
                    f_m_cnt++;

#if 1
                    double **para = new double *[4];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    double *res = new double[2];
                    f->Evaluate(para, res, NULL);
                    f_sum += sqrt(res[0] * res[0] + res[1] * res[1]);

                    double rho[3];
                    loss_function->Evaluate(res[0] * res[0] + res[1] * res[1], rho);
                    r_f_sum += rho[0];

//for (int j = 0; j < 2; j++)
//    sum_error_ceres += res[j];
//f->check(para);
#endif
                }
            }
        }
        ROS_INFO("visual measurement count: %d", f_m_cnt);
        ROS_INFO("visual measurement error: %f", f_sum / f_m_cnt);
        ROS_INFO("visual measurement r_err: %f", r_f_sum / f_m_cnt);
        sum_error_ceres += r_f_sum;
    }
    ROS_INFO("init error: %f", sum_error_ceres / 2);
    ROS_INFO("prepare for ceres: %f", t_prepare.toc());
    ROS_INFO("1 whole time for ceres: %f", t_whole.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::SPARSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    //options.use_explicit_schur_complement = true;
    options.minimizer_progress_to_stdout = true;
    //options.max_num_iterations = 10;
    //options.use_nonmonotonic_steps = true;
    options.max_solver_time_in_seconds = SOLVER_TIME;

    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.FullReport() << endl;
    ROS_INFO("solver costs: %f", t_solver.toc());
    ROS_INFO("imu: %f, projection: %f", IMUFactor_t::sum_t, ProjectionFactor::sum_t);
    ROS_INFO("2 whole time for ceres: %f", t_whole.toc());

    sum_error_ceres = 0.0;
    TicToc t_end;
    {
        double m_sum = 0.0;
        if (last_marginalization_factor)
        {
            double **para = new double *[last_marginalization_parameter_blocks.size()];
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                para[i] = last_marginalization_parameter_blocks[i];
            double *res = new double[last_marginalization_factor->num_residuals()];
            last_marginalization_factor->Evaluate(para, res, NULL);
            for (int i = 0; i < last_marginalization_factor->num_residuals(); i++)
            {
                m_sum += res[i] * res[i];
            }
        }
        ROS_INFO("marginalization error: %f", m_sum);
        sum_error_ceres += m_sum;
    }
    {
        double i_sum = 0.0;
        for (int i = 0; i < WINDOW_SIZE; i++)
        {
            int j = i + 1;

#if 1
            double **para = new double *[4];
            para[0] = para_Pose[i];
            para[1] = para_SpeedBias[i];
            para[2] = para_Pose[j];
            para[3] = para_SpeedBias[j];
            double *tmp_r = new double[15];
            imu_factors[j]->Evaluate(para, tmp_r, NULL);
            double tmp_sum = 0.0;

            for (int j = 0; j < 15; j++)
            {
                tmp_sum += tmp_r[j] * tmp_r[j];
                //printf("%f ", tmp_r[j] * tmp_r[j]);
            }
            //puts("");
            i_sum += tmp_sum;
//printf("dt: %f, %f\n", IMU_linear[j](6), tmp_sum);
//cout << IMU_linear[j].transpose() << endl;
//cout << imu_factors[j]->delta_p.transpose() << " " << imu_factors[j]->delta_v.transpose() << endl;
//imu_factors[j]->check(para);
#endif
        }
        ROS_INFO("imu error: %f", i_sum);
        sum_error_ceres += i_sum;
    }

    {
        int f_m_cnt = 0;
        double f_sum = 0.0;
        double r_f_sum = 0.0;
        int feature_index = -1;
        for (auto it_per_id : f_manager.traversal())
        {
            ++feature_index;

            int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
            int camera_i = it_per_id->feature_per_frame[0].feature_per_camera[0].camera_id;

            Vector3d pts_i = it_per_id->feature_per_frame[0].feature_per_camera[0].point;

            for (auto &it_per_frame : it_per_id->feature_per_frame)
            {
                imu_j++;
                for (auto &it_per_camera : it_per_frame.feature_per_camera)
                {
                    int camera_j = it_per_camera.camera_id;
                    if (imu_i == imu_j && camera_i == camera_j)
                    {
                        continue;
                    }
                    Vector3d pts_j = it_per_camera.point;
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                    f_m_cnt++;
#if 1
                    double **para = new double *[4];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    double *res = new double[2];
                    f->Evaluate(para, res, NULL);
                    f_sum += sqrt(res[0] * res[0] + res[1] * res[1]);
                    double rho[3];
                    loss_function->Evaluate(res[0] * res[0] + res[1] * res[1], rho);
                    r_f_sum += rho[0];
#endif
                }
            }
        }
        ROS_INFO("visual measurement count: %d", f_m_cnt);
        ROS_INFO("visual measurement error: %f", f_sum / f_m_cnt);
        ROS_INFO("visual measurement r_err: %f", r_f_sum / f_m_cnt);
        sum_error_ceres += r_f_sum;
    }
    ROS_INFO("final error: %f", sum_error_ceres / 2);
    ROS_INFO("end ceres costs: %f", t_end.toc());
    ROS_INFO("3 whole time for ceres: %f", t_whole.toc());

    new2old();

    vector<ceres::ResidualBlockId> residual_set;
    problem.GetResidualBlocks(&residual_set);
    for (auto it : residual_set)
        problem.RemoveResidualBlock(it);
    ROS_INFO("4 whole time for ceres: %f", t_whole.toc());

    TicToc t_whole_marginalization;
    if (!marginalization_flag)
    {
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor();
        old2new();
        if (last_marginalization_factor)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(last_marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_factor->addResidualBlockInfo(residual_block_info);
        }

        {
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factors[1], NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
            marginalization_factor->addResidualBlockInfo(residual_block_info);
        }

        {
            int feature_index = -1;
            for (auto it_per_id : f_manager.traversal())
            {
                ++feature_index;

                int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
                int camera_i = it_per_id->feature_per_frame[0].feature_per_camera[0].camera_id;

                Vector3d pts_i = it_per_id->feature_per_frame[0].feature_per_camera[0].point;

                for (auto &it_per_frame : it_per_id->feature_per_frame)
                {
                    imu_j++;
                    for (auto &it_per_camera : it_per_frame.feature_per_camera)
                    {
                        int camera_j = it_per_camera.camera_id;
                        if (imu_i == imu_j && camera_i == camera_j)
                            continue;
                        Vector3d pts_j = it_per_camera.point;
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);

                        if (imu_i == 0)
                        {
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                           vector<int>{0, 3});
                            marginalization_factor->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        ROS_INFO("begin pre marginalization");
        marginalization_factor->preMarginalize();
        ROS_INFO("end pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        ROS_INFO("begin marginalization");
        marginalization_factor->marginalize();
        ROS_INFO("end marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        vector<double *> parameter_blocks = marginalization_factor->getParameterBlocks(addr_shift);
        last_marginalization_factor = marginalization_factor;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        if (last_marginalization_factor &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationFactor *marginalization_factor = new MarginalizationFactor();
            old2new();
            if (last_marginalization_factor)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(last_marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_factor->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_INFO("begin marginalization");
            marginalization_factor->preMarginalize();
            ROS_INFO("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_INFO("begin marginalization");
            marginalization_factor->marginalize();
            ROS_INFO("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            vector<double *> parameter_blocks = marginalization_factor->getParameterBlocks(addr_shift);
            last_marginalization_factor = marginalization_factor;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
    ROS_INFO("whole marginalization costs: %f", t_whole_marginalization.toc());
    ROS_INFO("whole time for ceres: %f", t_whole.toc());
}

void SelfCalibrationEstimator::marginalize()
{
    TicToc t_margin;
    if (!marginalization_flag)
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                swap(use_cov[i], use_cov[i + 1]);
                IMU_linear[i].swap(IMU_linear[i + 1]);
                IMU_angular[i].swap(IMU_angular[i + 1]);
                Rs[i].swap(Rs[i + 1]);

                std::swap(imu_factors[i], imu_factors[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                IMU_cov[i].swap(IMU_cov[i + 1]);
                IMU_cov_nl[i].swap(IMU_cov_nl[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            use_cov[WINDOW_SIZE] = 1;

            delete imu_factors[WINDOW_SIZE];
            imu_factors[WINDOW_SIZE] = new IMUFactor_t{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            IMU_linear[WINDOW_SIZE].setZero();
            IMU_angular[WINDOW_SIZE].setIdentity();
            IMU_cov[WINDOW_SIZE].setZero();
            IMU_cov_nl[WINDOW_SIZE].setZero();

            marginalizeBack();
        }
        else
        {
            ROS_ASSERT(false);
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                propagateIMU(IMU_linear[frame_count - 1], IMU_angular[frame_count - 1],
                             IMU_cov[frame_count - 1], IMU_cov_nl[frame_count - 1],
                             tmp_dt, tmp_linear_acceleration, tmp_angular_velocity, Bas[frame_count - 2], Bgs[frame_count - 2]);

                imu_factors[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];
            use_cov[WINDOW_SIZE - 1]++;

            use_cov[WINDOW_SIZE] = 1;

            delete imu_factors[WINDOW_SIZE];
            imu_factors[WINDOW_SIZE] = new IMUFactor_t{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            IMU_linear[WINDOW_SIZE].setZero();
            IMU_angular[WINDOW_SIZE].setIdentity();
            IMU_cov[WINDOW_SIZE].setZero();
            IMU_cov_nl[WINDOW_SIZE].setZero();

            marginalizeFront();
        }
        else
        {
            ROS_ASSERT(false);
        }
    }
    ROS_DEBUG("marginalizetion costs: %f", t_margin.toc());
}

void SelfCalibrationEstimator::marginalizeFront()
{
    sum_of_front++;
    ROS_INFO("marginalize front");
    int n_calibration = NUM_OF_CAM * (solver_flag == LINEAR ? 3 : 6);
    int n_state = solver_flag == LINEAR ? 9 : 15;
    vector<int> remove, not_remove;
    for (int j = 0; j < n_state; j++)
        remove.push_back((frame_count - 1) * n_state + j);
    for (int i = 0; i < frame_count - 1; i++)
        for (int j = 0; j < n_state; j++)
            not_remove.push_back(i * n_state + j);
    for (int j = 0; j < n_state; j++)
        not_remove.push_back(frame_count * n_state + j);

    for (int j = 0; j < n_calibration; j++)
        not_remove.push_back((frame_count + 1) * n_state + j);

    f_manager.removeFront(frame_count, n_calibration, n_state, remove, not_remove);
    ROS_DEBUG("feature_count: %d\n", f_manager.getFeatureCount());
}

void SelfCalibrationEstimator::marginalizeBack()
{
    sum_of_back++;
    ROS_INFO("marginalize back");
    int n_calibration = NUM_OF_CAM * (solver_flag == LINEAR ? 3 : 6);
    int n_state = solver_flag == LINEAR ? 9 : 15;
    vector<int> remove, not_remove;
    for (int j = 0; j < n_state; j++)
        remove.push_back(j);
    for (int i = 1; i <= WINDOW_SIZE; i++)
        for (int j = 0; j < n_state; j++)
            not_remove.push_back(i * n_state + j);
    for (int j = 0; j < n_calibration; j++)
        not_remove.push_back((WINDOW_SIZE + 1) * n_state + j);

    ROS_DEBUG("feature_count: %d\n", f_manager.getFeatureCount());
    f_manager.removeBack(frame_count, n_calibration, n_state, remove, not_remove);
    ROS_DEBUG("feature_count: %d\n", f_manager.getFeatureCount());
}
