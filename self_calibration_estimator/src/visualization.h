#pragma once

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>

#include <eigen3/Eigen/Dense>

#include "self_calibration_estimator.h"

extern ros::Publisher pub_odometry, pub_odometry;
extern ros::Publisher pub_path, pub_pose;
extern ros::Publisher pub_cloud, pub_map;
extern ros::Publisher pub_init_guess, pub_key_poses;

extern ros::Publisher pub_ref_pose, pub_cur_pose;

extern ros::Publisher pub_key;
extern ros::Publisher pub_tic, pub_tic_cov;
extern ros::Publisher pub_ric, pub_ric_cov;
extern nav_msgs::Path path;

void registerPub(ros::NodeHandle &n);

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header);

void printStatistics(const SelfCalibrationEstimator &estimator, double t);

void pubOdometry(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header);

void pubInitialGuess(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header);

void pubKeyPoses(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header);

void pubCameraPose(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header);

void pubPointCloud(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header);

void pubCalibration(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header);

void pubTF(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header);
