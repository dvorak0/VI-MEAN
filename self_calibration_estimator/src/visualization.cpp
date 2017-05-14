#include "visualization.h"

using namespace ros;
using namespace Eigen;
ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path, pub_pose;
ros::Publisher pub_cloud, pub_map;
ros::Publisher pub_init_guess, pub_key_poses;

ros::Publisher pub_ref_pose, pub_cur_pose;

ros::Publisher pub_key;
ros::Publisher pub_tic, pub_tic_cov;
ros::Publisher pub_ric, pub_ric_cov;
nav_msgs::Path path;

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("latest_odometry", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_pose = n.advertise<geometry_msgs::PoseStamped>("pose", 1000);
    pub_cloud = n.advertise<sensor_msgs::PointCloud>("cloud", 1000);
    pub_map = n.advertise<sensor_msgs::PointCloud>("map", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_init_guess = n.advertise<visualization_msgs::Marker>("initial_guess", 1000);
    pub_key = n.advertise<geometry_msgs::PointStamped>("key", 1000);

    pub_ref_pose = n.advertise<geometry_msgs::PoseStamped>("ref_pose", 1000);
    pub_cur_pose = n.advertise<geometry_msgs::PoseStamped>("cur_pose", 1000);

    pub_tic = n.advertise<geometry_msgs::Vector3>("Pbc", 1000);
    pub_tic_cov = n.advertise<geometry_msgs::Vector3>("Pbc_cov", 1000);
    pub_ric = n.advertise<geometry_msgs::Vector3>("Rbc", 1000);
    pub_ric_cov = n.advertise<geometry_msgs::Vector3>("SVD", 1000);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, const std_msgs::Header &header)
{
    //ROS_ERROR("publish %f, at %f", header.stamp.toSec(), ros::Time::now().toSec());
    Eigen::Matrix3d R;
    R << 0, -1, 0,
        0, 0, -1,
        1, 0, 0;

    Eigen::Quaterniond quadrotor_Q = Q * Eigen::Quaterniond{R};

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "base";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = quadrotor_Q.x();
    odometry.pose.pose.orientation.y = quadrotor_Q.y();
    odometry.pose.pose.orientation.z = quadrotor_Q.z();
    odometry.pose.pose.orientation.w = quadrotor_Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry.publish(odometry);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;

    // body frame
    transform.setOrigin(tf::Vector3(P.x(),
                                    P.y(),
                                    P.z()));
    q.setW(Q.w());
    q.setX(Q.x());
    q.setY(Q.y());
    q.setZ(Q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "base", "latest"));

    // gal frame
    transform.setOrigin(tf::Vector3(0,
                                    0,
                                    0));
    q.setW(1);
    q.setX(0);
    q.setY(0);
    q.setZ(0);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "base", "map"));
}

void printStatistics(const SelfCalibrationEstimator &estimator, double t)
{
    ROS_INFO_STREAM("vo position: " << estimator.Ps[WINDOW_SIZE].transpose());
    ROS_INFO_STREAM("vo velocity: " << estimator.Vs[WINDOW_SIZE].transpose());
    ROS_INFO_STREAM("vo bias gyr: " << estimator.Bgs[WINDOW_SIZE].transpose());
    ROS_INFO_STREAM("vo bias acc: " << estimator.Bas[WINDOW_SIZE].transpose());
    ROS_INFO_STREAM("gravity: " << estimator.g.transpose() << " norm: " << estimator.g.norm());
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_INFO("calibration result for camera %d", i);
        ROS_INFO_STREAM("linear tic: " << estimator.last_tic[i].transpose());
        ROS_INFO_STREAM("linear ric: " << estimator.last_ric[i].transpose());

        ROS_INFO_STREAM("nonlinear tic: " << estimator.tic[i].transpose());
        ROS_INFO_STREAM("nonlinear ric: " << Utility::R2ypr(estimator.ric[i]).transpose());
        ROS_INFO_STREAM("covariance tic: " << estimator.tic_cov[i].transpose());
        ROS_INFO_STREAM("covariance ric: " << estimator.ric_cov[i].transpose());
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_INFO("vo solver costs: %f ms", t);
    ROS_INFO("average of time %f ms", sum_of_time / sum_of_calculation);

    static double sum_of_path = 0;
    static Vector3d last_path(0.0, 0.0, 0.0);
    sum_of_path += (estimator.Ps[WINDOW_SIZE] - last_path).norm();
    last_path = estimator.Ps[WINDOW_SIZE];
    ROS_INFO("sum of path %f", sum_of_path);
    ROS_INFO("end-to-end error: %f", estimator.Ps[WINDOW_SIZE].norm() / sum_of_path);
}

void pubOdometry(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header)
{
    if (estimator.solver_flag == SelfCalibrationEstimator::SolverFlag::NON_LINEAR)
    {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "base";
        odometry.child_frame_id = "base";
        odometry.pose.pose.position.x = estimator.Ps[WINDOW_SIZE].x();
        odometry.pose.pose.position.y = estimator.Ps[WINDOW_SIZE].y();
        odometry.pose.pose.position.z = estimator.Ps[WINDOW_SIZE].z();
        odometry.pose.pose.orientation.x = Quaterniond(estimator.Rs[WINDOW_SIZE]).x();
        odometry.pose.pose.orientation.y = Quaterniond(estimator.Rs[WINDOW_SIZE]).y();
        odometry.pose.pose.orientation.z = Quaterniond(estimator.Rs[WINDOW_SIZE]).z();
        odometry.pose.pose.orientation.w = Quaterniond(estimator.Rs[WINDOW_SIZE]).w();
        odometry.twist.twist.linear.x = estimator.Vs[WINDOW_SIZE](0);
        odometry.twist.twist.linear.y = estimator.Vs[WINDOW_SIZE](1);
        odometry.twist.twist.linear.z = estimator.Vs[WINDOW_SIZE](2);
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "base";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "base";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);
        pub_pose.publish(pose_stamped);
    }
}

void pubInitialGuess(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header)
{
    visualization_msgs::Marker init_guess;
    init_guess.header = header;
    init_guess.header.frame_id = "base";
    init_guess.ns = "initial_guess";
    init_guess.type = visualization_msgs::Marker::SPHERE;
    init_guess.action = visualization_msgs::Marker::ADD;
    init_guess.pose.orientation.w = 1.0;
    init_guess.pose.position.x = estimator.init_P.x();
    init_guess.pose.position.y = estimator.init_P.y();
    init_guess.pose.position.z = estimator.init_P.z();
    init_guess.lifetime = ros::Duration();

    init_guess.id = 0;
    init_guess.scale.x = 0.10;
    init_guess.scale.y = 0.10;
    init_guess.scale.z = 0.10;
    init_guess.color.g = 1.0;
    init_guess.color.a = 1.0;
    pub_init_guess.publish(init_guess);
}

void pubKeyPoses(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header)
{
    if (estimator.key_poses.size() == 0)
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "base";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.02;
    key_poses.scale.y = 0.02;
    key_poses.scale.z = 0.02;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        geometry_msgs::Point pose_marker;
        pose_marker.x = estimator.key_poses[i].x();
        pose_marker.y = estimator.key_poses[i].y();
        pose_marker.z = estimator.key_poses[i].z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header)
{
    int idx1 = WINDOW_SIZE - 2;
    int idx2 = WINDOW_SIZE - 1;
    if (estimator.solver_flag == SelfCalibrationEstimator::SolverFlag::NON_LINEAR)
    {
        {
            int i = idx1;
            geometry_msgs::PoseStamped camera_pose;
            camera_pose.header = header;
            camera_pose.header.frame_id = std::to_string(estimator.Headers[i].stamp.toNSec());
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);
            camera_pose.pose.position.x = P.x();
            camera_pose.pose.position.y = P.y();
            camera_pose.pose.position.z = P.z();
            camera_pose.pose.orientation.w = R.w();
            camera_pose.pose.orientation.x = R.x();
            camera_pose.pose.orientation.y = R.y();
            camera_pose.pose.orientation.z = R.z();

            pub_ref_pose.publish(camera_pose);
        }

        {
            int i = idx2;
            geometry_msgs::PoseStamped camera_pose;
            camera_pose.header = header;
            camera_pose.header.frame_id = std::to_string(estimator.Headers[i].stamp.toNSec());
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);
            camera_pose.pose.position.x = P.x();
            camera_pose.pose.position.y = P.y();
            camera_pose.pose.position.z = P.z();
            camera_pose.pose.orientation.w = R.w();
            camera_pose.pose.orientation.x = R.x();
            camera_pose.pose.orientation.y = R.y();
            camera_pose.pose.orientation.z = R.z();

            pub_cur_pose.publish(camera_pose);
        }
    }
}

void pubPointCloud(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud;
    point_cloud.header = header;
    for (auto &it : estimator.point_cloud)
    {
        geometry_msgs::Point32 p;
        p.x = it(0);
        p.y = it(1);
        p.z = it(2);
        point_cloud.points.push_back(p);
    }
    pub_cloud.publish(point_cloud);

    static sensor_msgs::PointCloud map_cloud;
    map_cloud.header = header;
    for (auto &it : estimator.margin_map)
    {
        geometry_msgs::Point32 p;
        p.x = it(0);
        p.y = it(1);
        p.z = it(2);
        map_cloud.points.push_back(p);
    }
    pub_map.publish(map_cloud);
}

void pubCalibration(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header)
{
    if (estimator.solver_flag == SelfCalibrationEstimator::SolverFlag::NON_LINEAR)
    {

        geometry_msgs::Vector3 tic;
        tic.x = estimator.tic[0](0);
        tic.y = estimator.tic[0](1);
        tic.z = estimator.tic[0](2);
        pub_tic.publish(tic);

        geometry_msgs::Vector3 tic_cov;
        tic_cov.x = estimator.tic_cov[0](0);
        tic_cov.y = estimator.tic_cov[0](1);
        tic_cov.z = estimator.tic_cov[0](2);
        pub_tic_cov.publish(tic_cov);

        geometry_msgs::Vector3 ric;
        ric.x = Utility::R2ypr(estimator.ric[0])(0);
        ric.y = Utility::R2ypr(estimator.ric[0])(1);
        ric.z = Utility::R2ypr(estimator.ric[0])(2);
        pub_ric.publish(ric);

        geometry_msgs::Vector3 ric_cov;
        ric_cov.x = estimator.ric_cov[0](0);
        ric_cov.y = estimator.ric_cov[0](1);
        ric_cov.z = estimator.ric_cov[0](2);
        pub_ric_cov.publish(ric_cov);
    }
}

void pubTF(const SelfCalibrationEstimator &estimator, const std_msgs::Header &header)
{
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;

    //Vector3d ng1 = estimator.g.normalized();
    //Vector3d ng2(0, 0, 1);
    //Quaterniond baseR = Quaterniond::FromTwoVectors(ng1, ng2).inverse();

    //// visualization frame
    //transform.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
    //q.setW(baseR.w());
    //q.setX(baseR.x());
    //q.setY(baseR.y());
    //q.setZ(baseR.z());

    //transform.setRotation(q);
    //br.sendTransform(tf::StampedTransform(transform, header.stamp, "base", "intermediate"));

    // body frame
    transform.setOrigin(tf::Vector3(estimator.Ps[WINDOW_SIZE - 2](0),
                                    estimator.Ps[WINDOW_SIZE - 2](1),
                                    estimator.Ps[WINDOW_SIZE - 2](2)));
    q.setW(Quaterniond(estimator.Rs[WINDOW_SIZE - 2]).w());
    q.setX(Quaterniond(estimator.Rs[WINDOW_SIZE - 2]).x());
    q.setY(Quaterniond(estimator.Rs[WINDOW_SIZE - 2]).y());
    q.setZ(Quaterniond(estimator.Rs[WINDOW_SIZE - 2]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "base", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                                    estimator.tic[0].y(),
                                    estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));
}
