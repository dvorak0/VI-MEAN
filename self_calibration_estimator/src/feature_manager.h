#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

class FeaturePerCamera
{
  public:
    FeaturePerCamera(int _camera_id, const Vector3d &_point)
        : camera_id(_camera_id)
    {
        z = _point(2);
        point = _point / z;
    }

    int camera_id;
    Vector3d point;
    double z;

    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

class FeaturePerFrame
{
  public:
    FeaturePerFrame(vector<FeaturePerCamera> &&_feature_per_camera)
        : feature_per_camera(std::forward<vector<FeaturePerCamera>>(_feature_per_camera))
    {
    }
    bool is_used;
    vector<FeaturePerCamera> feature_per_camera;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;

    int used_num;
    bool is_outlier;
    bool is_margin;
    double estimated_depth;

    Vector3d gt_p;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0)
    {
    }

    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Vector3d>>> &image, const map<int, Vector3d> &debug_image);
    void debugShow();
    void htmlShow();

    void shift(int n_start_frame);

    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count, int camera_id);

    vector<FeaturePerId *> traversal();
    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void tagMarginalizedPoints(bool is_non_linear, bool marginalization_flag);
    void removeBack(int frame_count, int n_calibration, int n_state, vector<int> &remove, vector<int> &not_remove);
    void removeFront(int frame_count, int n_calibration, int n_state, vector<int> &remove, vector<int> &not_remove);
    void removeOutlier();
    list<FeaturePerId> feature;

    std::vector<std::pair<int, std::vector<int>>> outlier_info;

  private:
    double compensatedParallax1(FeaturePerId &it_per_id);
    double compensatedParallax2(const FeaturePerId &it_per_id);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif
