#include "marginalization.h"

Marginalization::Marginalization() : number_imu(0), number_img(0)
{
}

VectorXd Marginalization::update(const MatrixXd &Ap, const VectorXd &bp,
                                 vector<int> &remove, const vector<int> &not_remove,
                                 int _start_imu, int _number_imu,
                                 int _start_img)
{
    ROS_DEBUG("Marginalization update");
    start_imu = _start_imu, number_imu = _number_imu;
    start_img = _start_img, number_img = not_remove.size() - number_imu;

    int l = bp.rows();
    int m = remove.size();
    int n = not_remove.size();
    ROS_DEBUG("l: %d, m: %d, n: %d", l, m, n);

    MatrixXd tmp_A(l, l);
    VectorXd tmp_x(l);

    n_Ap.resize(l, l);
    n_bp.resize(l);

    remove.insert(remove.end(), not_remove.begin(), not_remove.end());

    int j = 0;
    for (int i : remove)
    {
        tmp_A.row(j) = Ap.row(i);
        n_bp(j) = bp(i);
        j++;
    }

    j = 0;
    for (int i : remove)
    {
        n_Ap.col(j) = tmp_A.col(i);
        j++;
    }

    if (n != 0)
    {
        MatrixXd Amm = n_Ap.block(0, 0, m, m);
        Eigen::SelfAdjointEigenSolver<MatrixXd> saes(Amm);
        MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > 1e-8).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();

        VectorXd bmm = n_bp.segment(0, m);
        MatrixXd Amr = n_Ap.block(0, m, m, n);
        MatrixXd Arm = n_Ap.block(m, 0, n, m);
        MatrixXd Arr = n_Ap.block(m, m, n, n);
        VectorXd brr = n_bp.segment(m, n);
        n_Ap = Arr - Arm * Amm_inv * Amr;
        n_bp = brr - Arm * Amm_inv * bmm;
    }
    else
    {
        //todo
        MatrixXd Amm = n_Ap.block(0, 0, m, m);
        VectorXd bmm = n_bp.segment(0, m);
        n_Ap = Amm;
        n_bp = bmm;
    }
    return tmp_x.segment(m, n);
}

void Marginalization::setPrior(MatrixXd &A, VectorXd &b)
{
    if (number_imu != 0)
    {
        A.block(start_imu, start_imu, number_imu, number_imu) = n_Ap.block(0, 0, number_imu, number_imu);
        b.segment(start_imu, number_imu) = n_bp.segment(0, number_imu);

        if (number_img != 0)
        {
            A.block(start_imu, start_img, number_imu, number_img) = n_Ap.block(0, number_imu, number_imu, number_img);
            A.block(start_img, start_imu, number_img, number_imu) = n_Ap.block(number_imu, 0, number_img, number_imu);
            A.block(start_img, start_img, number_img, number_img) = n_Ap.block(number_imu, number_imu, number_img, number_img);
            b.segment(start_img, number_img) = n_bp.segment(number_imu, number_img);
        }
    }
}
