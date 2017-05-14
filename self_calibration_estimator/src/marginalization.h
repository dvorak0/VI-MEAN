#ifndef MARGINALIZATION_H
#define MARGINALIZATION_H

#include <vector>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "utility.h"

class Marginalization
{
public:
    Marginalization();
    VectorXd update(const MatrixXd &Ap, const VectorXd &bp,
                    vector<int> &remove, const vector<int> &not_remove,
                    int _start_imu, int _number_imu,
                    int _start_img);
    void setPrior(MatrixXd &A, VectorXd &b);
    MatrixXd n_Ap;
    VectorXd n_bp;
    int start_imu, number_imu;
    int start_img, number_img;
};

#endif
