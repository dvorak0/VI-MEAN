#pragma once

#include "integration_base.h"

class RK4Integration : public IntegrationBase<RK4Integration>
{
  public:
    RK4Integration(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                   const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : IntegrationBase<RK4Integration>{_acc_0, _gyr_0, _linearized_ba, _linearized_bg}
    {
    }

    template <typename Derived1, typename Derived2>
    void propagate_implementation(
        const Eigen::MatrixBase<Derived1> &tmp_delta_p, const Eigen::QuaternionBase<Derived2> &tmp_delta_q,
        const Eigen::MatrixBase<Derived1> &tmp_delta_v, const Eigen::MatrixBase<Derived1> &tmp_linearized_ba, const Eigen::MatrixBase<Derived1> &tmp_linearized_bg,
        const Eigen::MatrixBase<Derived1> &tmp_acc_0, const Eigen::MatrixBase<Derived1> &tmp_gyr_0, const Eigen::MatrixBase<Derived1> &tmp_acc_1, const Eigen::MatrixBase<Derived1> &tmp_gyr_1,
        Eigen::MatrixBase<Derived1> &delta_p, Eigen::QuaternionBase<Derived2> &delta_q,
        Eigen::MatrixBase<Derived1> &delta_v, Eigen::MatrixBase<Derived1> &linearized_ba, Eigen::MatrixBase<Derived1> &linearized_bg)
    {
        typedef typename Derived1::Scalar Scalar_t;

        Scalar_t dt_t(dt);

        Eigen::Matrix<Scalar_t, 3, 1> un_gyr_0 = tmp_gyr_0.template cast<Scalar_t>() - tmp_linearized_bg;
        Eigen::Matrix<Scalar_t, 3, 1> un_gyr_1 = tmp_gyr_1.template cast<Scalar_t>() - tmp_linearized_bg;
        Eigen::Matrix<Scalar_t, 3, 1> un_gyr_m = static_cast<Scalar_t>(0.5) * (un_gyr_0 + un_gyr_1);

        Eigen::Matrix<Scalar_t, 3, 1> un_acc_0 = tmp_acc_0.template cast<Scalar_t>() - tmp_linearized_ba;
        Eigen::Matrix<Scalar_t, 3, 1> un_acc_1 = tmp_acc_1.template cast<Scalar_t>() - tmp_linearized_ba;
        Eigen::Matrix<Scalar_t, 3, 1> un_acc_m = static_cast<Scalar_t>(0.5) * (un_acc_0 + un_acc_1);

        Eigen::Matrix<Scalar_t, 16, 1> y1;
        y1.template segment<3>(0) = tmp_delta_p;
        y1.template segment<4>(3) = Eigen::Matrix<Scalar_t, 4, 1>{tmp_delta_q.w(), tmp_delta_q.x(), tmp_delta_q.y(), tmp_delta_q.z()};
        y1.template segment<3>(7) = tmp_delta_v;
        y1.template segment<3>(10) = tmp_linearized_ba;
        y1.template segment<3>(13) = tmp_linearized_bg;

        Eigen::Matrix<Scalar_t, 16, 1> k1;
        k1.template segment<3>(0) = tmp_delta_v;
        k1.template segment<4>(3) = static_cast<Scalar_t>(0.5) * Utility::Qleft(tmp_delta_q) * Eigen::Matrix<Scalar_t, 4, 1>{static_cast<Scalar_t>(0.0), un_gyr_0.x(), un_gyr_0.y(), un_gyr_0.z()};
        k1.template segment<3>(7) = tmp_delta_q * un_acc_0;
        k1.template segment<3>(10) = Eigen::Matrix<Scalar_t, 3, 1>::Zero();
        k1.template segment<3>(13) = Eigen::Matrix<Scalar_t, 3, 1>::Zero();

        Eigen::Matrix<Scalar_t, 16, 1> y2 = y1 + static_cast<Scalar_t>(0.5) * dt_t * k1;
        Eigen::Matrix<Scalar_t, 16, 1> k2;
        Eigen::Quaternion<Scalar_t> delta_q_m = Eigen::Quaternion<Scalar_t>{y2(3), y2(4), y2(5), y2(6)};
        k2.template segment<3>(0) = y2.template segment<3>(7);
        k2.template segment<4>(3) = static_cast<Scalar_t>(0.5) * Utility::Qleft(delta_q_m) * Eigen::Matrix<Scalar_t, 4, 1>{static_cast<Scalar_t>(0.0), un_gyr_m.x(), un_gyr_m.y(), un_gyr_m.z()};
        k2.template segment<3>(7) = delta_q_m * un_acc_m;
        k2.template segment<3>(10) = Eigen::Matrix<Scalar_t, 3, 1>::Zero();
        k2.template segment<3>(13) = Eigen::Matrix<Scalar_t, 3, 1>::Zero();

        Eigen::Matrix<Scalar_t, 16, 1> y3 = y1 + static_cast<Scalar_t>(0.5) * dt_t * k2;
        Eigen::Matrix<Scalar_t, 16, 1> k3;
        delta_q_m = Eigen::Quaternion<Scalar_t>{y3(3), y3(4), y3(5), y3(6)};
        k3.template segment<3>(0) = y3.template segment<3>(7);
        k3.template segment<4>(3) = static_cast<Scalar_t>(0.5) * Utility::Qleft(delta_q_m) * Eigen::Matrix<Scalar_t, 4, 1>{static_cast<Scalar_t>(0.0), un_gyr_m.x(), un_gyr_m.y(), un_gyr_m.z()};
        k3.template segment<3>(7) = delta_q_m * un_acc_m;
        k3.template segment<3>(10) = Eigen::Matrix<Scalar_t, 3, 1>::Zero();
        k3.template segment<3>(13) = Eigen::Matrix<Scalar_t, 3, 1>::Zero();

        Eigen::Matrix<Scalar_t, 16, 1> y4 = y1 + dt_t * k3;
        Eigen::Matrix<Scalar_t, 16, 1> k4;
        delta_q_m = Eigen::Quaternion<Scalar_t>{y4(3), y4(4), y4(5), y4(6)};
        k4.template segment<3>(0) = y4.template segment<3>(7);
        k4.template segment<4>(3) = static_cast<Scalar_t>(0.5) * Utility::Qleft(delta_q_m) * Eigen::Matrix<Scalar_t, 4, 1>{static_cast<Scalar_t>(0.0), un_gyr_1.x(), un_gyr_1.y(), un_gyr_1.z()};
        k4.template segment<3>(7) = delta_q_m * un_acc_1;
        k4.template segment<3>(10) = Eigen::Matrix<Scalar_t, 3, 1>::Zero();
        k4.template segment<3>(13) = Eigen::Matrix<Scalar_t, 3, 1>::Zero();

        Eigen::Matrix<Scalar_t, 16, 1> y = y1 + static_cast<Scalar_t>(1.0 / 6.0) * dt_t * (k1 + static_cast<Scalar_t>(2) * k2 + static_cast<Scalar_t>(2) * k3 + k4);
        delta_p = y.template segment<3>(0);
        delta_q = Eigen::Quaternion<Scalar_t>{y(3), y(4), y(5), y(6)};
        delta_v = y.template segment<3>(7);
        linearized_ba = y.template segment<3>(10);
        linearized_bg = y.template segment<3>(13);
    }
};
