#include <glog/logging.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <diagnostic_msgs/KeyValue.h>
#include <execinfo.h>
#include <csignal>

#include "self_calibration_estimator.h"
#include "visualization.h"

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include "parameters.h"

SelfCalibrationEstimator estimator;

std::mutex m_buf;
std::condition_variable con;

double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> img_buf;

int sum_of_wait = 0;

std::mutex m_state;
double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba - tmp_Q.inverse() * estimator.g);

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba - tmp_Q.inverse() * estimator.g);

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = Eigen::Quaterniond{estimator.Rs[WINDOW_SIZE]};
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    double tmp_time = latest_time;
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

    ROS_INFO("prediction costs %f ms", t_predict.toc());

    ROS_ERROR("update delayed %f - %f = %f", latest_time, tmp_time, latest_time - tmp_time);
}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || img_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp > img_buf.front()->header.stamp))
        {
            ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp < img_buf.front()->header.stamp))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            img_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = img_buf.front();
        img_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp <= img_msg->header.stamp)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "base";
        pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
    }
}

std::map<long long, cv::Mat> image_pool;
void raw_image_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImagePtr img_ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    image_pool[img_msg->header.stamp.toNSec()] = img_ptr->image;
}

std::map<long long, sensor_msgs::PointCloudConstPtr> cloud_pool;
void image_callback(const sensor_msgs::PointCloudConstPtr &img_msg)
{
    m_buf.lock();
    img_buf.push(img_msg);
    m_buf.unlock();
    con.notify_one();

    cloud_pool[img_msg->header.stamp.toNSec()] = img_msg;
}

void send_imu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (current_time < 0)
        current_time = t;
    double dt = t - current_time;
    current_time = t;

#ifdef GT
    double ba[]{0.0, 0.0, 0.0};
    double bg[]{0.0, 0.0, 0.0};
#else
    double ba[]{0.0, 0.0, 0.0};
    double bg[]{0.0, 0.0, 0.0};
#endif

    double dx = imu_msg->linear_acceleration.x - ba[0];
    double dy = imu_msg->linear_acceleration.y - ba[1];
    double dz = imu_msg->linear_acceleration.z - ba[2];

    double rx = imu_msg->angular_velocity.x - bg[0];
    double ry = imu_msg->angular_velocity.y - bg[1];
    double rz = imu_msg->angular_velocity.z - bg[2];
    ROS_INFO("IMU %f, dt: %f, acc: %f %f %f, gyr: %f %f %f", t, dt, dx, dy, dz, rx, ry, rz);

    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
}

void process()
{

    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });

        ROS_INFO("got measurement of size: %lu", measurements.size());

        lk.unlock();

        for (auto &measurement : measurements)
        {
            for (auto &imu_msg : measurement.first)
                send_imu(imu_msg);

            auto img_msg = measurement.second;

            ROS_INFO("begin: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());

            ROS_INFO("processing vision data with stamp %f", img_msg->header.stamp.toSec());

            TicToc t_s;
            map<int, vector<pair<int, Vector3d>>> image;
            map<int, Vector3d> debug_image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                ROS_ASSERT(z == 1);
                if (img_msg->channels.size() == 5)
                {
                    debug_image[feature_id] = Vector3d(img_msg->channels[2].values[i],
                                                       img_msg->channels[3].values[i],
                                                       img_msg->channels[4].values[i]);
                }
                image[feature_id].emplace_back(camera_id, Vector3d(x, y, z));
            }

            estimator.processImage(image, img_msg->header, debug_image);
            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);

            std_msgs::Header header = img_msg->header;
            header.frame_id = "base";
            pubOdometry(estimator, header);
            pubInitialGuess(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubCalibration(estimator, header);
            pubTF(estimator, header);
            ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }

        m_buf.lock();
        m_state.lock();
        update();
        m_state.unlock();
        m_buf.unlock();

        ROS_INFO("\n");
    }
}

void debug_callback(const diagnostic_msgs::KeyValueConstPtr &key_value_msg)
{
    std::cout << key_value_msg->key << std::endl;
    std::cout << key_value_msg->value << std::endl;
    int pid = std::stoi(key_value_msg->value);

    for (auto it_per_id : estimator.f_manager.traversal())
        if (it_per_id->feature_id == pid)
            printf("%d %f\n", pid, it_per_id->estimated_depth);

    for (auto i : estimator.outlier_info)
    {
        printf("%d : ", i.first);
        int name_id = 1;
        for (auto j : i.second)
        {
            cv::Mat img = image_pool[j].clone();
            sensor_msgs::PointCloudConstPtr feature = cloud_pool[j];
            int idx = -1;
            float u, v;
            for (int k = 0; k < static_cast<int>(feature->channels[0].values.size()); k++)
            {
                if (int(feature->channels[0].values[k] + 0.5) == i.first)
                {
                    idx = k;
                    u = feature->channels[1].values[k];
                    v = feature->channels[2].values[k];
                    break;
                }
            }
            ROS_ASSERT(idx != -1);
            printf("(%f,%f) ", u, v);
            cv::circle(img, cv::Point2f(u, v), 2, cv::Scalar(0), 2);
            cv::imshow(std::to_string(name_id++), img);
        }
        puts("");
    }
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    ros::init(argc, argv, "self_calibration_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setIMUModel();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_INFO("EIGEN_DONT_PARALLELIZE");
#endif

    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe("imu", 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("image", 2000, image_callback);
    ros::Subscriber sub_debug = n.subscribe("debug", 2000, debug_callback);
    ros::Subscriber sub_raw_image = n.subscribe("raw_image", 2000, raw_image_callback);

    std::thread loop{process};

    ros::spin();

    return 0;
}
