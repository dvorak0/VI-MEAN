#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>

#include <tuple>
#include <memory>
#include <iostream>

#include "imu_factor.h"
#include "euler_integration.h"
#include "midpoint_integration.h"
#include "RK4_integration.h"

static int const MAX_TIME = 10;
static int const MAX_BOX = 10;

//const double ACC_N = 0.1;
//const double GYR_N = 0.05;
//const double ACC_W = 0.002;
//const double GYR_W = 4.0e-5;

#define Y_COS 2
#define Z_COS 2 * 2

Eigen::Vector3d getPosition(double t)
{
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

    return Eigen::Vector3d{x, y, z};
}

Eigen::Matrix3d getRotation(double t)
{
    return (Eigen::AngleAxisd{30.0 / 180 * M_PI * sin(t / MAX_TIME * M_PI * 2), Eigen::Vector3d::UnitX()} *
            Eigen::AngleAxisd{40.0 / 180 * M_PI * sin(t / MAX_TIME * M_PI * 2), Eigen::Vector3d::UnitY()} *
            Eigen::AngleAxisd{0, Eigen::Vector3d::UnitZ()}).toRotationMatrix();
}

Eigen::Vector3d getAngularVelocity(double t)
{
    const double delta_t = 1.0e-6;
    Eigen::Matrix3d rot = getRotation(t);
    t += delta_t;
    Eigen::Matrix3d drot = (getRotation(t) - rot) / delta_t;
    t -= delta_t;
    Eigen::Matrix3d skew = rot.inverse() * drot;
    return Eigen::Vector3d{skew(2, 1), -skew(2, 0), skew(1, 0)};
}

Eigen::Vector3d getVelocity(double t)
{
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

    return Eigen::Vector3d{dx, dy, dz};
}

Eigen::Vector3d getLinearAcceleration(double t)
{
    double ddx, ddy, ddz;
    if (t < MAX_TIME)
    {
        ddx = MAX_BOX / 2.0 * -cos(t / MAX_TIME * M_PI) * (1.0 / MAX_TIME * M_PI) * (1.0 / MAX_TIME * M_PI);
        ddy = MAX_BOX / 2.0 * -cos(t / MAX_TIME * M_PI * Y_COS) * (1.0 / MAX_TIME * M_PI * Y_COS) * (1.0 / MAX_TIME * M_PI * Y_COS);
        ddz = MAX_BOX / 2.0 * -cos(t / MAX_TIME * M_PI * Z_COS) * (1.0 / MAX_TIME * M_PI * Z_COS) * (1.0 / MAX_TIME * M_PI * Z_COS);
    }
    else if (t >= MAX_TIME && t < 2 * MAX_TIME)
    {
        ddx = 0.0;
        ddy = 0.0;
        ddz = 0.0;
    }
    else
    {
        double tt = t - 2 * MAX_TIME;
        ddx = MAX_BOX / 2.0 * -cos(tt / MAX_TIME * M_PI) * (1.0 / MAX_TIME * M_PI) * (1.0 / MAX_TIME * M_PI);
        ddy = MAX_BOX / 2.0 * -cos(tt / MAX_TIME * M_PI * Y_COS) * (1.0 / MAX_TIME * M_PI * Y_COS) * (1.0 / MAX_TIME * M_PI * Y_COS);
        ddz = MAX_BOX / 2.0 * -cos(tt / MAX_TIME * M_PI * Z_COS) * (1.0 / MAX_TIME * M_PI * Z_COS) * (1.0 / MAX_TIME * M_PI * Z_COS);
    }
    return getRotation(t).inverse() * (Eigen::Vector3d{ddx, ddy, ddz} + G);
}

template <class Integration>
std::tuple<double, double, double> test_function(int T, int FREQ, double Ti, double Tj)
{
    typedef IMUFactor<Integration> IMUFactor_t;

    Eigen::Vector3d acc_0 = getLinearAcceleration(Ti);
    Eigen::Vector3d gyr_0 = getAngularVelocity(Ti);

    std::shared_ptr<IMUFactor_t> imu_factor(new IMUFactor_t{acc_0, gyr_0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()});

    double dt = 1.0 / FREQ;
    for (int i = 1; i <= T; i++)
    {
        double t = Ti + dt * i;
        Eigen::Vector3d acc = getLinearAcceleration(t);
        Eigen::Vector3d gyr = getAngularVelocity(t);
        imu_factor->push_back(dt, acc, gyr);
    }

    double **para = new double *[4];
    para[0] = new double[7];
    para[1] = new double[9];
    para[2] = new double[7];
    para[3] = new double[9];

    Eigen::Map<Eigen::Vector3d>{para[0]} = getPosition(Ti);
    Eigen::Map<Eigen::Quaterniond>{para[0] + 3} = Eigen::Quaterniond{getRotation(Ti)};
    Eigen::Map<Eigen::Vector3d>{para[1]} = getVelocity(Ti);
    Eigen::Map<Eigen::Matrix<double, 6, 1>>{para[1] + 3}.setZero();

    Eigen::Map<Eigen::Vector3d>{para[2]} = getPosition(Tj);
    Eigen::Map<Eigen::Quaterniond>{para[2] + 3} = Eigen::Quaterniond{getRotation(Tj)};
    Eigen::Map<Eigen::Vector3d>{para[3]} = getVelocity(Tj);
    Eigen::Map<Eigen::Matrix<double, 6, 1>>{para[3] + 3}.setZero();

    double *res = new double[15];
    imu_factor->Evaluate(para, res, NULL);

    double e_p = sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2]);
    double e_q = sqrt(0.25 * (res[3] * res[3] + res[4] * res[4] + res[5] * res[5]));
    double e_v = sqrt(res[6] * res[6] + res[7] * res[7] + res[8] * res[8]);
    return std::make_tuple(e_p, 2.0 * asin(e_q), e_v);
}

void print_stat(char n, const std::vector<double> &d)
{
    printf("%c min: %f, max: %f, avg: %f\n", n,
           *std::min_element(std::begin(d), std::end(d)),
           *std::max_element(std::begin(d), std::end(d)),
           std::accumulate(std::begin(d), std::end(d), 0.0) / d.size());
}

template <class Integration>
void test_loop(int T, int FREQ)
{
    const double t = 1.0 * T / FREQ;
    std::vector<double> e_ps, e_qs, e_vs;
    for (int i = 0; i < 5; i++)
    {
        double Ti = t * i, Tj = t * i + t;
        std::tuple<double, double, double> e = test_function<Integration>(T, FREQ, Ti, Tj);
        e_ps.push_back(std::get<0>(e));
        e_qs.push_back(std::get<1>(e));
        e_vs.push_back(std::get<2>(e));
    }
    print_stat('p', e_ps);
    print_stat('q', e_qs);
    print_stat('v', e_vs);
}

TEST(TestSuite, 200Euler)
{
    test_loop<EulerIntegration>(400, 200);
}

TEST(TestSuite, 200Midpoint)
{
    test_loop<MidpointIntegration>(400, 200);
}

TEST(TestSuite, 200RK4)
{
    test_loop<RK4Integration>(400, 200);
}

TEST(TestSuite, 500Euler)
{
    test_loop<EulerIntegration>(1000, 500);
}

TEST(TestSuite, 500Midpoint)
{
    test_loop<MidpointIntegration>(1000, 500);
}

TEST(TestSuite, 500RK4)
{
    test_loop<RK4Integration>(1000, 500);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);

    ACC_N = 0.1;
    GYR_N = 0.05;
    ACC_W = 0.002;
    GYR_W = 4.0e-5;

    return RUN_ALL_TESTS();
}
