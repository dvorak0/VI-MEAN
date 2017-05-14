#pragma once

#include "integration_base.h"

class MidpointIntegration : public IntegrationBase<MidpointIntegration>
{
  public:
    MidpointIntegration(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                        const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
        : IntegrationBase<MidpointIntegration>{_acc_0, _gyr_0, _linearized_ba, _linearized_bg}
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

        Eigen::Matrix<Scalar_t, 3, 1> un_acc_0 = tmp_delta_q * (tmp_acc_0.template cast<Scalar_t>() - tmp_linearized_ba);

        Eigen::Matrix<Scalar_t, 3, 1> un_gyr = static_cast<Scalar_t>(0.5) * (tmp_gyr_0.template cast<Scalar_t>() + tmp_gyr_1.template cast<Scalar_t>()) - tmp_linearized_bg;
        delta_q = tmp_delta_q * Utility::deltaQ(un_gyr * dt_t);

        Eigen::Matrix<Scalar_t, 3, 1> un_acc_1 = delta_q * (tmp_acc_1.template cast<Scalar_t>() - tmp_linearized_ba);

        Eigen::Matrix<Scalar_t, 3, 1> un_acc = static_cast<Scalar_t>(0.5) * (un_acc_0 + un_acc_1);

        delta_p = tmp_delta_p + dt_t * tmp_delta_v + static_cast<Scalar_t>(0.5) * dt_t * dt_t * un_acc;
        delta_v = tmp_delta_v + dt_t * un_acc;

        linearized_ba = tmp_linearized_ba;
        linearized_bg = tmp_linearized_bg;
    }
};
