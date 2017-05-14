#pragma once

#include "utility.h"
#include "parameters.h"

#include <ceres/ceres.h>

template <class Derived>
class IntegrationBase
{
  public:
    IntegrationBase() = delete;
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg) : linearized_acc{_acc_0}, linearized_gyr{_gyr_0}
    {
        init_linearization_point(_linearized_ba, _linearized_bg);
    }

    void init_linearization_point(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        linearized_ba = _linearized_ba;
        linearized_bg = _linearized_bg;
        m_jacobian.setIdentity();
        m_covariance.setZero();

        has_jaco = false;
        jacobian.setIdentity();
        covariance.setZero();

        sum_dt = 0.0;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
    }

    void push_back(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt_buf.push_back(_dt);
        acc_buf.push_back(_acc_1);
        gyr_buf.push_back(_gyr_1);
        propagate(_dt, _acc_1, _gyr_1);
    }

    void repropagate(const std::vector<double> &dt_buf, const std::vector<Eigen::Vector3d> &acc_buf, const std::vector<Eigen::Vector3d> &gyr_buf)
    {
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    struct functor
    {
        functor(Derived *_derived) : derived{_derived} {}

        template <typename T>
        bool operator()(const T *const p, const T *const q, const T *const v, const T *const ba, const T *const bg,
                        const T *const a_0, const T *const g_0, const T *const a_1, const T *const g_1, T *e) const
        {
            Eigen::Matrix<T, 3, 1> tmp_delta_p{p[0], p[1], p[2]};
            Eigen::Quaternion<T> tmp_delta_q{q[0], q[1], q[2], q[3]};
            Eigen::Matrix<T, 3, 1> tmp_delta_v{v[0], v[1], v[2]};
            Eigen::Matrix<T, 3, 1> tmp_linearized_ba{ba[0], ba[1], ba[2]};
            Eigen::Matrix<T, 3, 1> tmp_linearized_bg{bg[0], bg[1], bg[2]};

            Eigen::Matrix<T, 3, 1> tmp_acc_0{a_0[0], a_0[1], a_0[2]};
            Eigen::Matrix<T, 3, 1> tmp_gyr_0{g_0[0], g_0[1], g_0[2]};

            Eigen::Matrix<T, 3, 1> tmp_acc_1{a_1[0], a_1[1], a_1[2]};
            Eigen::Matrix<T, 3, 1> tmp_gyr_1{g_1[0], g_1[1], g_1[2]};

            Eigen::Matrix<T, 3, 1> delta_p;
            Eigen::Quaternion<T> delta_q;
            Eigen::Matrix<T, 3, 1> delta_v;
            Eigen::Matrix<T, 3, 1> linearized_ba;
            Eigen::Matrix<T, 3, 1> linearized_bg;

            derived->propagate_implementation(tmp_delta_p, tmp_delta_q, tmp_delta_v, tmp_linearized_ba, tmp_linearized_bg,
                                              tmp_acc_0, tmp_gyr_0, tmp_acc_1, tmp_gyr_1,
                                              delta_p, delta_q, delta_v, linearized_ba, linearized_bg);

            Eigen::Map<Eigen::Matrix<T, 3, 1>>(e + 0) = delta_p;
            Eigen::Map<Eigen::Matrix<T, 4, 1>>(e + 3) = Eigen::Matrix<T, 4, 1>{delta_q.w(), delta_q.x(), delta_q.y(), delta_q.z()};
            Eigen::Map<Eigen::Matrix<T, 3, 1>>(e + 7) = delta_v;
            Eigen::Map<Eigen::Matrix<T, 3, 1>>(e + 10) = linearized_ba;
            Eigen::Map<Eigen::Matrix<T, 3, 1>>(e + 13) = linearized_bg;

            return true;
        }
        Derived *derived;
    };

    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
    {
        dt = _dt;
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;

        Derived *derived = static_cast<Derived *>(this);
        const Eigen::Vector3d tmp_delta_p = delta_p;
        const Eigen::Quaterniond tmp_delta_q = delta_q;
        const Eigen::Vector3d tmp_delta_v = delta_v;
        const Eigen::Vector3d tmp_linearized_ba = linearized_ba;
        const Eigen::Vector3d tmp_linearized_bg = linearized_bg;

        const Eigen::Vector3d tmp_acc_0 = acc_0;
        const Eigen::Vector3d tmp_gyr_0 = gyr_0;
        const Eigen::Vector3d tmp_acc_1 = acc_1;
        const Eigen::Vector3d tmp_gyr_1 = gyr_1;

        //std::cout << delta_q.coeffs().transpose() << std::endl;

        //derived->propagate_implementation(tmp_delta_p, tmp_delta_q, tmp_delta_v, tmp_linearized_ba, tmp_linearized_bg,
        //                                  delta_p, delta_q, delta_v, linearized_ba, linearized_bg);

        ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<functor, 16, 3, 4, 3, 3, 3, 3, 3, 3, 3>(new functor(derived));

        Eigen::Matrix<double, 4, 1> tmp_q{tmp_delta_q.w(), tmp_delta_q.x(), tmp_delta_q.y(), tmp_delta_q.z()};
        const std::vector<const double *> parameters{tmp_delta_p.data(), tmp_q.data(), tmp_delta_v.data(), tmp_linearized_ba.data(), tmp_linearized_bg.data(),
                                                     tmp_acc_0.data(), tmp_gyr_0.data(), tmp_acc_1.data(), tmp_gyr_1.data()};

        Eigen::Matrix<double, 16, 1> residuals;

        Eigen::Matrix<double, 16, 3, Eigen::RowMajor> de_dp;
        Eigen::Matrix<double, 16, 4, Eigen::RowMajor> de_dq;
        Eigen::Matrix<double, 16, 3, Eigen::RowMajor> de_dv;
        Eigen::Matrix<double, 16, 3, Eigen::RowMajor> de_dba;
        Eigen::Matrix<double, 16, 3, Eigen::RowMajor> de_dbg;

        Eigen::Matrix<double, 16, 3, Eigen::RowMajor> de_da0;
        Eigen::Matrix<double, 16, 3, Eigen::RowMajor> de_dg0;
        Eigen::Matrix<double, 16, 3, Eigen::RowMajor> de_da1;
        Eigen::Matrix<double, 16, 3, Eigen::RowMajor> de_dg1;

        std::vector<double *> jacobians{de_dp.data(), de_dq.data(), de_dv.data(), de_dba.data(), de_dbg.data(),
                                        de_da0.data(), de_dg0.data(), de_da1.data(), de_dg1.data()};

        cost_function->Evaluate(parameters.data(), residuals.data(), jacobians.data());

        delta_p = residuals.segment<3>(0);
        delta_q = Eigen::Quaterniond(residuals(3), residuals(4), residuals(5), residuals(6));
        delta_v = residuals.segment<3>(7);
        linearized_ba = residuals.segment<3>(10);
        linearized_bg = residuals.segment<3>(13);

        Eigen::Matrix<double, 16, 16> A;
        A.middleCols<3>(0) = de_dp;
        A.middleCols<4>(3) = de_dq;
        A.middleCols<3>(7) = de_dv;
        A.middleCols<3>(10) = de_dba;
        A.middleCols<3>(13) = de_dbg;

        Eigen::Matrix<double, 16, 12> B;
        B.middleCols<3>(0) = de_da0;
        B.middleCols<3>(3) = de_dg0;
        B.middleCols<3>(6) = de_da1;
        B.middleCols<3>(9) = de_dg1;

        Eigen::Matrix<double, 12, 12> noise = Eigen::Matrix<double, 12, 12>::Zero();
        noise.block<3, 3>(0, 0) = noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity() / dt;
        noise.block<3, 3>(3, 3) = noise.block<3, 3>(9, 9) = (ACC_N * GYR_N) * Eigen::Matrix3d::Identity() / dt;

        Eigen::Matrix<double, 16, 6> C = Eigen::Matrix<double, 16, 6>::Zero();
        C.block<3, 3>(10, 0) = Eigen::Matrix3d::Identity();
        C.block<3, 3>(13, 3) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 6, 6> walk = Eigen::Matrix<double, 6, 6>::Zero();
        walk.block<3, 3>(0, 0) = dt * ACC_W * ACC_W * Eigen::Matrix3d::Identity();
        walk.block<3, 3>(3, 3) = dt * GYR_W * GYR_W * Eigen::Matrix3d::Identity();

        m_jacobian = A * m_jacobian;
        m_covariance = A * m_covariance * A.transpose() + B * noise * B.transpose() + C * walk * C.transpose();

        has_jaco = false;

        delta_q.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;
    }

    void getJacobianAndCovariance()
    {
        if (!has_jaco)
        {
            Eigen::Matrix<double, 15, 16> L = Eigen::Matrix<double, 15, 16>::Zero();
            L.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            L.block<3, 4>(3, 3) = 2 * Utility::Qleft(delta_q.inverse()).template bottomRows<3>();
            L.block<3, 3>(6, 7) = Eigen::Matrix3d::Identity();
            L.block<3, 3>(9, 10) = Eigen::Matrix3d::Identity();
            L.block<3, 3>(12, 13) = Eigen::Matrix3d::Identity();

            Eigen::Matrix<double, 16, 15> R = Eigen::Matrix<double, 16, 15>::Zero();
            R.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            R.block<4, 3>(3, 3) = 0.5 * Utility::Qleft(Eigen::Quaterniond::Identity()).template rightCols<3>();
            R.block<3, 3>(7, 6) = Eigen::Matrix3d::Identity();
            R.block<3, 3>(10, 9) = Eigen::Matrix3d::Identity();
            R.block<3, 3>(13, 12) = Eigen::Matrix3d::Identity();

            jacobian = L * m_jacobian * R;
            covariance = L * m_covariance * L.transpose();

            has_jaco = true;
            //std::cout << covariance << std::endl
            //          << std::endl;
            //std::cout << m_covariance << std::endl
            //          << std::endl;
        }
    }

    Eigen::Matrix<double, 15, 1> evaluate(
        const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
        const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::Vector3d dba = Bai - linearized_ba;
        Eigen::Vector3d dbg = Bgi - linearized_bg;

        if (dba.norm() > BIAS_ACC_THRESHOLD || dbg.norm() > BIAS_GYR_THRESHOLD)
        {
            init_linearization_point(Bai, Bgi);
            repropagate(dt_buf, acc_buf, gyr_buf);
            dba = Bai - linearized_ba;
            dbg = Bgi - linearized_bg;
        }

        getJacobianAndCovariance();
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix<double, 16, 16> m_jacobian, m_covariance;

    bool has_jaco;
    Eigen::Matrix<double, 15, 15> jacobian, covariance;

    double sum_dt;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;

    //static Eigen::Matrix<double, 12, 12> NOISE;
};

//template <typename T>
//Eigen::Matrix<double, 12, 12> IntegrationBase<T>::NOISE;
