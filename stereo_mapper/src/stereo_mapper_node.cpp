#include <cstdio>
#include <cstring>
#include <map>

#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <stereo_msgs/DisparityImage.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>

#include <Eigen/Dense>
#include <opencv/cxeigen.hpp>

#include <dynamic_reconfigure/server.h>
#include <stereo_mapper/debugConfig.h>

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "stereo_mapper.h"

#define STEREO 0
#define NORMALIZE 0

bool start = false;
StereoMapper mapper;

cv::Mat show_img;
int debug_param1;
int debug_param2;
int debug_param3;
int debug_param4;
int debug_param5;
FILE *f_out;

ros::Publisher pub_point_cloud2;
ros::Publisher pub_point_cloud_ref;
ros::Publisher pub_color_img, pub_disp_img, pub_color_img_info, pub_disp_img_info;

Eigen::Matrix3d K_eigen;
Eigen::Matrix3d R_bs;
Eigen::Vector3d P_bs;
cv::Mat T_BS;

cv::Mat K1, D1;
cv::Mat K2, D2;
cv::Mat R21, T21;
cv::Mat R1, P1, R2, P2, Q;
cv::Mat map11, map12;
cv::Mat map21, map22;
std::set<int> id_set_l;
int sta_cnt[10][64 + 10][64 + 10];
bool feature_ratio(const std::vector<float> &id_set_r)
{
    double r = 0.0;
    if (id_set_l.size() != 0)
    {
        double cnt = 0;
        for (auto id : id_set_r)
            cnt += id_set_l.find(id + 0.5) != id_set_l.end();
        r = cnt / id_set_l.size();
    }

    if (r < 0.8)
    {
        id_set_l.clear();
        for (auto id : id_set_r)
            id_set_l.insert(id + 0.5);
        ROS_INFO("init keyframe with ratio: %f", r);
        return true;
    }
    else
    {
        ROS_INFO("temporally update keyframe with ratio: %f", r);
        return false;
    }
}

geometry_msgs::PoseStampedConstPtr key_pose;
void writePose2Cloud(const sensor_msgs::PointCloud2Ptr &cloud_ptr, const geometry_msgs::PoseStampedConstPtr &pose_ptr)
{
    double *addr = reinterpret_cast<double *>(&(cloud_ptr->data[0]));
    memcpy(addr++, &(pose_ptr->pose.position.x), sizeof(double));
    memcpy(addr++, &(pose_ptr->pose.position.y), sizeof(double));
    memcpy(addr++, &(pose_ptr->pose.position.z), sizeof(double));
    memcpy(addr++, &(pose_ptr->pose.orientation.w), sizeof(double));
    memcpy(addr++, &(pose_ptr->pose.orientation.x), sizeof(double));
    memcpy(addr++, &(pose_ptr->pose.orientation.y), sizeof(double));
    memcpy(addr++, &(pose_ptr->pose.orientation.z), sizeof(double));
}

// void writePose2Cloud(const sensor_msgs::PointCloud2Ptr &cloud_ptr, const std::shared_ptr<Data> &data_ptr)
//{
//    double *addr = reinterpret_cast<double *>(&(cloud_ptr->data[0]));
//    memcpy(addr++, &(data_ptr->R.w()), sizeof(double));
//    memcpy(addr++, &(data_ptr->R.x()), sizeof(double));
//    memcpy(addr++, &(data_ptr->R.y()), sizeof(double));
//    memcpy(addr++, &(data_ptr->R.z()), sizeof(double));
//
//    memcpy(addr++, &(data_ptr->P.x()), sizeof(double));
//    memcpy(addr++, &(data_ptr->P.y()), sizeof(double));
//    memcpy(addr++, &(data_ptr->P.z()), sizeof(double));
//}

std_msgs::Header key_header;
void sendCloud2(const cv::Mat &dense_points_, const cv::Mat &un_img_l0)
{
    sensor_msgs::PointCloud2Ptr points(new sensor_msgs::PointCloud2);
    points->header = key_header;
    points->header.frame_id = "ref_frame";

    points->height = dense_points_.rows;
    points->width = dense_points_.cols;
    points->fields.resize(4);
    points->fields[0].name = "x";
    points->fields[0].offset = 0;
    points->fields[0].count = 1;
    points->fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    points->fields[1].name = "y";
    points->fields[1].offset = 4;
    points->fields[1].count = 1;
    points->fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    points->fields[2].name = "z";
    points->fields[2].offset = 8;
    points->fields[2].count = 1;
    points->fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    points->fields[3].name = "rgb";
    points->fields[3].offset = 12;
    points->fields[3].count = 1;
    points->fields[3].datatype = sensor_msgs::PointField::FLOAT32;
    //points.is_bigendian = false; ???
    points->point_step = 16;
    points->row_step = points->point_step * points->width;
    points->data.resize(points->row_step * points->height);
    points->is_dense = false; // there may be invalid points

    float bad_point = std::numeric_limits<float>::quiet_NaN();
    int i = 0;
    for (int32_t u = 0; u < dense_points_.rows; ++u)
    {
        for (int32_t v = 0; v < dense_points_.cols; ++v, ++i)
        {
            cv::Vec3f p = dense_points_.at<cv::Vec3f>(u, v);
            float dep = p[2];
            float x = p[0];
            float y = p[1];
            if (dep < 100.0f)
            {
                uint8_t g = un_img_l0.at<uint8_t>(u, v);
                int32_t rgb = (g << 16) | (g << 8) | g;
                memcpy(&points->data[i * points->point_step + 0], &x, sizeof(float));
                memcpy(&points->data[i * points->point_step + 4], &y, sizeof(float));
                memcpy(&points->data[i * points->point_step + 8], &dep, sizeof(float));
                memcpy(&points->data[i * points->point_step + 12], &rgb, sizeof(int32_t));
            }
            else
            {
                memcpy(&points->data[i * points->point_step + 0], &bad_point, sizeof(float));
                memcpy(&points->data[i * points->point_step + 4], &bad_point, sizeof(float));
                memcpy(&points->data[i * points->point_step + 8], &bad_point, sizeof(float));
                memcpy(&points->data[i * points->point_step + 12], &bad_point, sizeof(float));
            }
        }
    }
    pub_point_cloud_ref.publish(points);
}
void sendCloud(const cv::Mat &dense_points_, const cv::Mat &un_img_l0)
{
    sensor_msgs::PointCloud2Ptr points(new sensor_msgs::PointCloud2);
    points->header = key_header;
    points->header.frame_id = "ref_frame";

    points->height = dense_points_.rows;
    points->width = dense_points_.cols;
    points->fields.resize(4);
    points->fields[0].name = "x";
    points->fields[0].offset = 0;
    points->fields[0].count = 1;
    points->fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    points->fields[1].name = "y";
    points->fields[1].offset = 4;
    points->fields[1].count = 1;
    points->fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    points->fields[2].name = "z";
    points->fields[2].offset = 8;
    points->fields[2].count = 1;
    points->fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    points->fields[3].name = "rgb";
    points->fields[3].offset = 12;
    points->fields[3].count = 1;
    points->fields[3].datatype = sensor_msgs::PointField::FLOAT32;
    //points.is_bigendian = false; ???
    points->point_step = 16;
    points->row_step = points->point_step * points->width;
    points->data.resize(points->row_step * points->height);
    points->is_dense = false; // there may be invalid points

    float bad_point = std::numeric_limits<float>::quiet_NaN();
    int i = 0;
    for (int32_t u = 0; u < dense_points_.rows; ++u)
    {
        for (int32_t v = 0; v < dense_points_.cols; ++v, ++i)
        {
            float dep = dense_points_.at<float>(u, v);
#if DOWNSAMPLE
            float x = dep * (v - K1.at<double>(0, 2) / 2) / (K1.at<double>(0, 0) / 2);
            float y = dep * (u - K1.at<double>(1, 2) / 2) / (K1.at<double>(1, 1) / 2);
#else
            float x = dep * (v - K1.at<double>(0, 2)) / K1.at<double>(0, 0);
            float y = dep * (u - K1.at<double>(1, 2)) / K1.at<double>(1, 1);
#endif
            if (dep < DEP_INF)
            {
                uint8_t g = un_img_l0.at<uint8_t>(u, v);
                int32_t rgb = (g << 16) | (g << 8) | g;
                memcpy(&points->data[i * points->point_step + 0], &x, sizeof(float));
                memcpy(&points->data[i * points->point_step + 4], &y, sizeof(float));
                memcpy(&points->data[i * points->point_step + 8], &dep, sizeof(float));
                memcpy(&points->data[i * points->point_step + 12], &rgb, sizeof(int32_t));
            }
            else
            {
                memcpy(&points->data[i * points->point_step + 0], &bad_point, sizeof(float));
                memcpy(&points->data[i * points->point_step + 4], &bad_point, sizeof(float));
                memcpy(&points->data[i * points->point_step + 8], &bad_point, sizeof(float));
                memcpy(&points->data[i * points->point_step + 12], &bad_point, sizeof(float));
            }
        }
    }
    pub_point_cloud2.publish(points);
}
int image_id = 0;

char str_calib[100], str_result[100];
char str_obj[100];
const char *path_calib = "/home/yzf/data/kitti/data_scene_flow_calib/training/calib";
const char *path_result = "/home/yzf/data/kitti/results/ps_ad/data/disp_0";
const char *path_obj = "/home/yzf/data/kitti/data_scene_flow/training/obj_map";
cv::Mat K;

cv::Mat img1, img2;
Eigen::Matrix3d R1_eigen, R2_eigen;
Eigen::Vector3d T1_eigen, T2_eigen;
cv::Mat draw_img;

cv::Mat showColorDep(const cv::Mat &result)
{
    cv::Mat color_result(result.rows, result.cols, CV_8UC3);
    float map[8][4] = {{0, 0, 0, 114}, {0, 0, 1, 185}, {1, 0, 0, 114}, {1, 0, 1, 174}, {0, 1, 0, 114}, {0, 1, 1, 185}, {1, 1, 0, 114}, {1, 1, 1, 0}};

    float sum = 0;
    for (int32_t i = 0; i < 8; i++)
        sum += map[i][3];

    float weights[8]; // relative weights
    float cumsum[8];  // cumulative weights
    cumsum[0] = 0;
    for (int32_t i = 0; i < 7; i++)
    {
        weights[i] = sum / map[i][3];
        cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
    }

    float max_disp = 0.0f;
    for (int32_t v = 0; v < result.rows; v++)
        for (int32_t u = 0; u < result.cols; u++)
            if (result.at<float>(v, u) > max_disp)
                max_disp = result.at<float>(v, u);

    for (int32_t v = 0; v < result.rows; v++)
    {
        for (int32_t u = 0; u < result.cols; u++)
        {

            // get normalized value
            float val = std::min(std::max(result.at<float>(v, u) / max_disp, 0.0f), 1.0f);

            // find bin
            int32_t i;
            for (i = 0; i < 7; i++)
                if (val < cumsum[i + 1])
                    break;

            // compute red/green/blue values
            float w = 1.0 - (val - cumsum[i]) * weights[i];
            uint8_t r = (uint8_t)((w * map[i][0] + (1.0 - w) * map[i + 1][0]) * 255.0);
            uint8_t g = (uint8_t)((w * map[i][1] + (1.0 - w) * map[i + 1][1]) * 255.0);
            uint8_t b = (uint8_t)((w * map[i][2] + (1.0 - w) * map[i + 1][2]) * 255.0);

            // set pixel
            cv::Vec3b bgr;
            bgr.val[0] = b;
            bgr.val[1] = g;
            bgr.val[2] = r;
            color_result.at<cv::Vec3b>(v, u) = bgr;
        }
    }
    return color_result;
}

//void callback_img(
//    const sensor_msgs::ImageConstPtr &img_l_msg,
//    const sensor_msgs::ImageConstPtr &img_r_msg,
//    const sensor_msgs::PointCloudConstPtr &img_f_msg,
//    const geometry_msgs::PoseStampedConstPtr &pose_msg)
//{
//    puts("callback");
//    cv_bridge::CvImagePtr img_l_ptr = cv_bridge::toCvCopy(img_l_msg, sensor_msgs::image_encodings::MONO8);
//    cv_bridge::CvImagePtr img_r_ptr = cv_bridge::toCvCopy(img_r_msg, sensor_msgs::image_encodings::MONO8);
//    //cv::imshow("left", img_l_ptr->image);
//    //cv::imshow("right", img_r_ptr->image);
//
//    Eigen::Vector3d T = Eigen::Vector3d{pose_msg->pose.position.x,
//                                        pose_msg->pose.position.y,
//                                        pose_msg->pose.position.z};
//
//    Eigen::Matrix3d R = Eigen::Quaterniond{pose_msg->pose.orientation.w,
//                                           pose_msg->pose.orientation.x,
//                                           pose_msg->pose.orientation.y,
//                                           pose_msg->pose.orientation.z}.toRotationMatrix();
//
//    cv::Mat cv_R, cv_T;
//    cv::eigen2cv(R, cv_R);
//    cv::eigen2cv(T, cv_T);
//
//    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
//    if (feature_ratio(img_f_msg->channels[0].values))
//    {
//        start = true;
//        cv::undistort(img_l_ptr->image, img1, K1, D1);
//
//        key_header = pose_msg->header;
//        key_pose = pose_msg;
//
//        mapper.initIntrinsic(K1, D1, K2, D2);
//
//#if NORMALIZE
//        cv::Mat img_ll;
//        clahe->apply(img_l_ptr->image, img_ll);
//        mapper.initReference(img_ll, cv_R, cv_T);
//#else
//        mapper.initReference(img_l_ptr->image, cv_R, cv_T);
//#endif
//
//        cv::imshow("ep line", img1);
//        cv::cv2eigen(cv_R, R1_eigen);
//        cv::cv2eigen(cv_T, T1_eigen);
//
//        //int64 t = cv::getTickCount();
//        //sprintf(str_obj, "%s/%06d_10.png", path_obj, image_id);
//        //cv::Mat img_obj = cv::imread(str_obj);
//        //cv::Mat img_obj2 = img_obj.clone();
//        //cv::cvtColor(img_obj, img_obj, CV_BGR2GRAY);
//        //cv::cvtColor(img_obj2, img_obj2, CV_BGR2GRAY);
//
//        //for (int u = 1; u < img_obj.rows - 1; u++)
//        //    for (int v = 1; v < img_obj.cols - 1; v++)
//        //        if (img_obj.at<uchar>(u, v) ||
//        //            img_obj.at<uchar>(u - 1, v) ||
//        //            img_obj.at<uchar>(u, v - 1) ||
//        //            img_obj.at<uchar>(u, v + 1) ||
//        //            img_obj.at<uchar>(u + 1, v))
//        //            img_obj2.at<uchar>(u, v) = 255;
//        //mapper.setMask(img_obj2);
//        //printf("mask: %fms\n", (cv::getTickCount() - t) * 1000 / cv::getTickFrequency());
//        ////cv::imshow("obj", img_obj2);
//
//        //// cv_R = cv::Mat::eye(3, 3, CV_64F);
//        //// cv_T = cv::Mat::zeros(3, 1, CV_64F), cv_T.at<double>(0, 0) = BASE_LINE;
//        //cv::Mat result = mapper.update(img_r_ptr->image, cv_R, cv_T);
//        //cv::Mat show_result;
//        //result.convertTo(show_result, CV_16U, 256);
//
//        //cv::cv2eigen(cv_R, R2_eigen);
//        //cv::cv2eigen(cv_T, T2_eigen);
//        //img2 = img_r_ptr->image.clone();
//        //draw_img = result;
//
//        //cv::imshow("result", show_result);
//        //sprintf(str_result, "%s/%06d_10.png", path_result, image_id);
//        //cv::imwrite(str_result, show_result);
//        //image_id++;
//
//        //// cv_R = cv::Mat::eye(3, 3, CV_64F);
//        //// cv_T = cv::Mat::zeros(3, 1, CV_64F);
//        //// cv_T.at<double>(0, 0) = 0.54;
//    }
//    else
//    {
//        start = true;
//        ROS_INFO("update");
//        cv::undistort(img_l_ptr->image, img2, K1, D1);
//
//#if NORMALIZE
//        cv::Mat img_ll;
//        clahe->apply(img_l_ptr->image, img_ll);
//        mapper.update(img_ll, cv_R, cv_T);
//#else
//        mapper.update(img_l_ptr->image, cv_R, cv_T);
//#endif
//        cv::Mat result = mapper.output();
//
//        cv::Mat_<cv::Vec3f> dense_points_(HEIGHT, WIDTH);
//        for (int i = 0; i < HEIGHT; i++)
//            for (int j = 0; j < WIDTH; j++)
//            {
//                float z = result.at<float>(i, j);
//                dense_points_(i, j)[0] = (j - K1.at<double>(0, 2)) / K1.at<double>(0, 0) * z;
//                dense_points_(i, j)[1] = (i - K1.at<double>(1, 2)) / K1.at<double>(1, 1) * z;
//                dense_points_(i, j)[2] = z;
//            }
//        sendCloud(dense_points_, img1);
//
//        {
//            cv_bridge::CvImage out_msg;
//            out_msg.header = img_l_msg->header;
//            out_msg.header.frame_id = "camera";
//            out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
//            out_msg.image = result.clone();
//
//            pub_disp_img.publish(out_msg.toImageMsg());
//        }
//
//        {
//            cv_bridge::CvImage out_msg;
//            out_msg.header = img_l_msg->header;
//            out_msg.header.frame_id = "camera";
//            out_msg.encoding = sensor_msgs::image_encodings::MONO8;
//            out_msg.image = img1.clone();
//            pub_color_img.publish(out_msg.toImageMsg());
//        }
//
//        {
//            sensor_msgs::CameraInfo camera_info;
//            camera_info.header = img_l_msg->header;
//            camera_info.P[0] = K1.at<double>(0, 0);
//            camera_info.P[5] = K1.at<double>(1, 1);
//            camera_info.P[2] = K1.at<double>(0, 2);
//            camera_info.P[6] = K1.at<double>(1, 2);
//            camera_info.width = WIDTH;
//            camera_info.height = HEIGHT;
//            pub_color_img_info.publish(camera_info);
//            pub_disp_img_info.publish(camera_info);
//        }
//
//        {
//            // camera frame
//            static tf::TransformBroadcaster br;
//            tf::Transform transform;
//            tf::Quaternion q;
//            transform.setOrigin(tf::Vector3(T1_eigen.x(),
//                                            T1_eigen.y(),
//                                            T1_eigen.z()));
//            q.setW(Eigen::Quaterniond(R1_eigen).w());
//            q.setX(Eigen::Quaterniond(R1_eigen).x());
//            q.setY(Eigen::Quaterniond(R1_eigen).y());
//            q.setZ(Eigen::Quaterniond(R1_eigen).z());
//            transform.setRotation(q);
//            br.sendTransform(tf::StampedTransform(transform, img_l_msg->header.stamp, "base", "ref_frame"));
//        }
//
//        cv::cv2eigen(cv_R, R2_eigen);
//        cv::cv2eigen(cv_T, T2_eigen);
//        cv::Mat show_result;
//        result.convertTo(show_result, CV_16U, 256);
//
//        cv::Mat color_result = showColorDep(result);
//        //cv::imshow("result", show_result);
//        cv::imshow("color_result", color_result);
//        //sprintf(str_result, "%s/%06d_10.png", path_result, image_id);
//        //cv::imwrite(str_result, show_result);
//        image_id++;
//    }
//}

cv::StereoBM bm(cv::StereoBM::BASIC_PRESET, 128, 21);

cv::Mat blockMatching(const std::string name, const cv::Mat &img_l, const cv::Mat &img_r)
{
    ROS_INFO("bm");

    TicToc t_bm;
    cv::Mat img_ll, img_rr;
    cv::remap(img_l, img_ll, map11, map12, cv::INTER_LINEAR);
    cv::remap(img_r, img_rr, map21, map22, cv::INTER_LINEAR);
    char str[100];

    sprintf(str, "/home/yzf/code/stereo_benchmark/images/%s_l.png", name.c_str());
    cv::imwrite(std::string(str), img_ll);

    sprintf(str, "/home/yzf/code/stereo_benchmark/images/%s_r.png", name.c_str());
    cv::imwrite(std::string(str), img_rr);

    cv::Mat disp_16, disp;
    cv::Mat_<cv::Vec3f> dense_points_;
    bm(img_ll, img_rr, disp_16);
    disp_16.convertTo(disp, CV_32F, 1.0f / 16);
    cv::reprojectImageTo3D(disp, dense_points_, Q, true);
    sendCloud2(dense_points_, img_ll);

    //cv::imwrite("/home/ubuntu/disp_bm.jpg", disp_8);
    return disp;
}

static void onMouse(int event, int x, int y, int, void *object)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        printf("click x: %d, y: %d\n", x, y);

        for (int k = 0; k < DEP_CNT; k++)
        {
            float idep = k * DEP_SAMPLE;
            float dep = 1. / idep;
            Eigen::Vector3d pts_src = Eigen::Vector3d(x, y, 1) * dep;
            Eigen::Vector3d pts_dst = K_eigen * (R2_eigen.transpose() * (R1_eigen * K_eigen.inverse() * pts_src + T1_eigen - T2_eigen));
            cv::circle(img2, cv::Point2f(pts_dst.x() / pts_dst.z(), pts_dst.y() / pts_dst.z()), 1, cv::Scalar(-1));
            printf("%f %f\n", pts_dst.x() / pts_dst.z(), pts_dst.y() / pts_dst.z());
        }
        //float idep = draw_img.at<float>(y, x) * DEP_SAMPLE;
        //float dep = 1.0f / idep;
        //Eigen::Vector3d pts_src = Eigen::Vector3d(x, y, 1) * dep;
        //Eigen::Vector3d pts_dst = K_eigen * (R2_eigen.transpose() * (R1_eigen * K_eigen.inverse() * pts_src + T1_eigen - T2_eigen));
        //cv::circle(img2, cv::Point2f(pts_dst.x() / pts_dst.z(), pts_dst.y() / pts_dst.z()), 3, cv::Scalar(-1));
        //cv::imshow("debug2", img2);
        //cv::circle(img2, cv::Point2f(pts_dst.x() / pts_dst.z(), pts_dst.y() / pts_dst.z()), 1, cv::Scalar(-1));
    }
}

std::map<std::string, cv::Mat> img_pool;
std::map<std::string, cv::Mat> img_pool_r;
#if BENCHMARK
std::map<std::string, cv::Mat> img_pool_disp;
#endif

void callback_raw_image(
    const sensor_msgs::ImageConstPtr &img_l_msg,
    const sensor_msgs::ImageConstPtr &img_r_msg)
{
    ROS_INFO("save images with header: %f", img_l_msg->header.stamp.toSec());
    cv_bridge::CvImagePtr img_l_ptr = cv_bridge::toCvCopy(img_l_msg, sensor_msgs::image_encodings::MONO8);
    img_pool[std::to_string(img_l_ptr->header.stamp.toNSec())] = img_l_ptr->image.clone();
    cv_bridge::CvImagePtr img_r_ptr = cv_bridge::toCvCopy(img_r_msg, sensor_msgs::image_encodings::MONO8);
    img_pool_r[std::to_string(img_r_ptr->header.stamp.toNSec())] = img_r_ptr->image.clone();
    char str[100];
    std::string name = std::to_string(img_l_ptr->header.stamp.toNSec());

#if OFFLINE
    cv::Mat img_ll, img_rr;
    cv::remap(img_l_ptr->image, img_ll, map11, map12, cv::INTER_LINEAR);
    cv::remap(img_r_ptr->image, img_rr, map21, map22, cv::INTER_LINEAR);

    sprintf(str, "/home/yzf/code/stereo_benchmark/images/%s_l.png", name.c_str());
    cv::imwrite(str, img_ll);
    sprintf(str, "/home/yzf/code/stereo_benchmark/images/%s_r.png", name.c_str());
    cv::imwrite(str, img_rr);
#endif

#if BENCHMARK
    sprintf(str, "/home/yzf/code/stereo_benchmark/15_fast/%s.bin", name.c_str());
    printf("reading %s\n", str);
    int fd = open(str, O_RDONLY);
    float *d = static_cast<float *>(mmap(NULL, 480 * 752 * sizeof(float), PROT_READ, MAP_SHARED, fd, 0));
    cv::Mat disp{480, 752, CV_32F};
    for (int u = 0; u < 480; u++)
        for (int v = 0; v < 752; v++)
            disp.at<float>(u, v) = d[u * 752 + v];
    close(fd);

    img_pool_disp[std::to_string(img_r_ptr->header.stamp.toNSec())] = disp;

#endif
}

std::string last_time;
cv::Mat result;

void callback_raw_pose(const geometry_msgs::PoseStampedConstPtr &ref_pose_ptr,
                       const geometry_msgs::PoseStampedConstPtr &cur_pose_ptr)
{
    double t = clock();
    std::string ref_time = ref_pose_ptr->header.frame_id;
    std::string cur_time = cur_pose_ptr->header.frame_id;
    std::map<std::string, cv::Mat>::iterator ref_it = img_pool.find(ref_time);
    std::map<std::string, cv::Mat>::iterator cur_it = img_pool.find(cur_time);
    //img_pool.erase(img_pool.begin(), ref_it);
    if (ref_it != img_pool.end() && cur_it != img_pool.end())
    {
        ROS_INFO_STREAM("ref_time: " << ref_time << " cur_time: " << cur_time);

#if BENCHMARK
        std::map<std::string, cv::Mat>::iterator ref_it_r = img_pool_r.find(ref_time);
        //img_pool_r.erase(img_pool_r.begin(), ref_it_r);
        std::map<std::string, cv::Mat>::iterator ref_it_disp = img_pool_disp.find(ref_time);
        cv::Mat disp = ref_it_disp->second.clone();
        cv::Mat bm_result;
        cv::reprojectImageTo3D(disp, bm_result, Q, true);
#endif
        Eigen::Matrix3d R_l = Eigen::Quaterniond{ref_pose_ptr->pose.orientation.w,
                                                 ref_pose_ptr->pose.orientation.x,
                                                 ref_pose_ptr->pose.orientation.y,
                                                 ref_pose_ptr->pose.orientation.z}.toRotationMatrix();
        Eigen::Vector3d T_l = Eigen::Vector3d{ref_pose_ptr->pose.position.x,
                                              ref_pose_ptr->pose.position.y,
                                              ref_pose_ptr->pose.position.z};

        cv::Mat cv_R_l, cv_T_l;
        cv::eigen2cv(R_l, cv_R_l);
        cv::eigen2cv(T_l, cv_T_l);

        Eigen::Matrix3d R_r = Eigen::Quaterniond{cur_pose_ptr->pose.orientation.w,
                                                 cur_pose_ptr->pose.orientation.x,
                                                 cur_pose_ptr->pose.orientation.y,
                                                 cur_pose_ptr->pose.orientation.z}.toRotationMatrix();
        Eigen::Vector3d T_r = Eigen::Vector3d{cur_pose_ptr->pose.position.x,
                                              cur_pose_ptr->pose.position.y,
                                              cur_pose_ptr->pose.position.z};

        cv::Mat cv_R_r, cv_T_r;
        cv::eigen2cv(R_r, cv_R_r);
        cv::eigen2cv(T_r, cv_T_r);

        if (last_time != ref_time)
        {

            last_time = ref_time;

            mapper.initReference(ref_it->second);
            img1 = mapper.img_intensity;

            //cv::imshow("ep line", img1);
            cv::cv2eigen(cv_R_l, R1_eigen);
            cv::cv2eigen(cv_T_l, T1_eigen);
        }

        {
            // camera frame
            static tf::TransformBroadcaster br;
            tf::Transform transform;
            tf::Quaternion q;
            transform.setOrigin(tf::Vector3(T_l.x(),
                                            T_l.y(),
                                            T_l.z()));
            q.setW(Eigen::Quaterniond(R_l).w());
            q.setX(Eigen::Quaterniond(R_l).x());
            q.setY(Eigen::Quaterniond(R_l).y());
            q.setZ(Eigen::Quaterniond(R_l).z());
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform, ref_pose_ptr->header.stamp, "base", "ref_frame"));
            key_header = ref_pose_ptr->header;
        }

        start = true;

        //cv::undistort(cur_it->second, img2, K1, D1);
        cv::cv2eigen(cv_R_r, R2_eigen);
        cv::cv2eigen(cv_T_r, T2_eigen);

        //cv::imshow("ep line", img1);
        double t_cpu = clock();
        double t_wall = cv::getTickCount();

        TicToc t_update;
        mapper.update(cur_it->second, cv_R_l, cv_T_l, cv_R_r, cv_T_r);
        ROS_INFO("update costs: %fms", t_update.toc());

        ROS_INFO("cpu time: %fms, wall time: %fms",
                 (clock() - t_cpu) / CLOCKS_PER_SEC * 1000,
                 (cv::getTickCount() - t_wall) / cv::getTickFrequency());

        result = mapper.output();
        //cv::medianBlur(tmp, result, 3);
        //puts("remove all points");
        //for (int32_t u = 0; u < HEIGHT; ++u)
        //    for (int32_t v = 0; v < WIDTH; ++v)
        //    {
        //        float depth = result.at<float>(u, v);
        //        if (depth < 1.0f)
        //            result.at<float>(u, v) = 1000.0f;
        //    }
#if 0
        {
            cv::Mat disp2{HEIGHT, WIDTH, CV_32F, cv::Scalar(0.0)};
            for (int32_t u = 0; u < HEIGHT; ++u)
                for (int32_t v = 0; v < WIDTH; ++v)
                {
                    float depth = result.at<float>(u, v);
                    float p = 1.0f / depth / DEP_SAMPLE;
                    int pn = p + 0.5f;
                    disp2.at<float>(u, v) = p;
                }

            double s = 6;
            cv::Mat scale_disp;
            disp2.convertTo(scale_disp, CV_8U, s * 2);
            cv::Mat full = scale_disp.clone();
            full = cv::Scalar(255);
            cv::Mat color_disp;
            cv::applyColorMap(full - scale_disp, color_disp, cv::COLORMAP_JET);
            cv::imshow("motion", color_disp);
            cv::waitKey(10);
        }
#endif

#if BENCHMARK
        cv::Mat disp2{HEIGHT, WIDTH, CV_32F, cv::Scalar(0.0)};

        int a1 = 0, b1 = 0;
        int a2 = 0, b2 = 0;
        int a3 = 0, b3 = 0;
        float c3 = 0.0f;
        int cnt = 0;
        for (int32_t u = 0; u < HEIGHT; ++u)
            for (int32_t v = 0; v < WIDTH; ++v)
            {
                float depth = result.at<float>(u, v);
                if (depth < 100.0f)
                {
                    float p = 1.0f / depth / DEP_SAMPLE;
                    int pn = p + 0.5f;
                    disp2.at<float>(u, v) = p;

                    float r_depth = bm_result.at<cv::Vec3f>(u * 2, v * 2)[2];
                    if (r_depth < 100.0f)
                    {
                        float r_p = 1.0f / r_depth / DEP_SAMPLE;
                        int r_pn = r_p + 0.5f;

                        sta_cnt[0][pn][r_pn]++;
                        if (fabs(p - r_p) <= 1)
                            a1++;
                        else
                            b1++;
                        if (fabs(p - r_p) <= 2)
                            a2++;
                        else
                            b2++;
                        if (fabs(p - r_p) <= 3)
                        {
                            a3++;
                            c3 += fabs(p - r_p);
                        }
                        else
                            b3++;
                    }
                }
            }
        double s = 6;
        {
            cv::Mat scale_disp;
            disp.convertTo(scale_disp, CV_8U, s);
            cv::Mat full = scale_disp.clone();
            full = cv::Scalar(255);
            cv::Mat color_disp;
            cv::applyColorMap(full - scale_disp, color_disp, cv::COLORMAP_JET);
            cv::Mat color_disp_resize;
            cv::resize(color_disp, color_disp_resize, cv::Size(WIDTH, HEIGHT));
            cv::imshow("spatial stereo", color_disp_resize);

            cv::imwrite("/home/yzf/ref.png", color_disp);
        }

        {
            cv::Mat scale_disp;
            disp2.convertTo(scale_disp, CV_8U, s * 2);
            cv::Mat full = scale_disp.clone();
            full = cv::Scalar(255);
            cv::Mat color_disp;
            cv::applyColorMap(full - scale_disp, color_disp, cv::COLORMAP_JET);
            cv::imshow("motion stereo", color_disp);
            static int sum_cnt = 0;
            char path[100];
            sprintf(path, "/home/yzf/motion%04d_%02d.png", sum_cnt++, mapper.measurement_cnt);
            cv::imwrite(path, color_disp);
            cv::imwrite("/home/yzf/raw.png", mapper.img_intensity);
        }
        cv::imshow("reference", mapper.img_intensity);
        cv::imshow("measurement", mapper.img_intensity_r);
        cv::waitKey(10);

        //for (int i = 0; i < 64; i++)
        //{
        //    for (int j = 0; j < 64; j++)
        //    {
        //        printf("%d ", sta_cnt[0][i][j]);
        //    }
        //    puts("");
        //}
        //fprintf(f_out, "%d %d %d %d %d %d %d %f\n", mapper.measurement_cnt, a1, b1, a2, b2, a3, b3, c3);
        fprintf(f_out, "%f %f %f %f %f %f %f %f %f\n", std::stod(ref_time) / 1e9, 1.0 * mapper.measurement_cnt,
                1.0 * a1, 1.0 * b1, 1.0 * a2, 1.0 * b2, 1.0 * a3, 1.0 * b3, c3 / (a3 + b3));
        fflush(f_out);
#endif

        ROS_INFO("publish to Fusion: %f", key_header.stamp.toSec());
        sendCloud(result, img1);

        ROS_INFO("publish point cloud: %f", key_header.stamp.toSec());
        {
            cv_bridge::CvImage out_msg;
            out_msg.header = key_header;
            out_msg.header.frame_id = "camera";
            out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            out_msg.image = result.clone();
            pub_disp_img.publish(out_msg.toImageMsg());
        }

        {
            cv_bridge::CvImage out_msg;
            out_msg.header = key_header;
            out_msg.header.frame_id = "camera";
            out_msg.encoding = sensor_msgs::image_encodings::MONO8;
            out_msg.image = img1.clone();
            pub_color_img.publish(out_msg.toImageMsg());
        }

        {
            sensor_msgs::CameraInfo camera_info;
            camera_info.header = key_header;
#if DOWNSAMPLE
            camera_info.P[0] = K1.at<double>(0, 0) / 2;
            camera_info.P[5] = K1.at<double>(1, 1) / 2;
            camera_info.P[2] = K1.at<double>(0, 2) / 2;
            camera_info.P[6] = K1.at<double>(1, 2) / 2;
#else
            camera_info.P[0] = K1.at<double>(0, 0);
            camera_info.P[5] = K1.at<double>(1, 1);
            camera_info.P[2] = K1.at<double>(0, 2);
            camera_info.P[6] = K1.at<double>(1, 2);
#endif
            camera_info.width = WIDTH;
            camera_info.height = HEIGHT;
            pub_color_img_info.publish(camera_info);
            pub_disp_img_info.publish(camera_info);
        }
        ROS_INFO("CPU time costs: %f", (clock() - t) / CLOCKS_PER_SEC);
    }
}

void drawAndShow(std::string name, const cv::Mat &img, const std::vector<cv::Point2f> &pts)
{
    cv::Mat img_ = img.clone();
    for (int i = 0; i < static_cast<int>(pts.size()); i++)
    {
        cv::circle(img_, pts[i], 3, cv::Scalar(-1));
        char name[10];
        sprintf(name, "%d", i);
        cv::putText(img_, name, pts[i], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    //cv::imshow(name, img_);
}

void callback_debug(stereo_mapper::debugConfig &config, uint32_t level)
{
    if (!start)
        return;
    pi1 = config.pi1;
    pi2 = config.pi2;
    tau_so = config.tau_so;
    sgm_q1 = config.sgm_q1;
    sgm_q2 = config.sgm_q2;
    sgm_iter = config.sgm_iter;
    var_scale = 0.1f * config.var_scale;
    var_width = config.var_width;
    ROS_INFO("pi1 %f", pi1);
    ROS_INFO("pi2 %f", pi2);
    ROS_INFO("tau_so %f", tau_so);
    ROS_INFO("sgm_q1 %f", sgm_q1);
    ROS_INFO("sgm_q2 %f", sgm_q2);
    ROS_INFO("sgm_iter %d", sgm_iter);
    ROS_INFO("var_scale %f", var_scale);
    ROS_INFO("var_width %d", var_width);
    ROS_INFO("x %f", config.x);
    ROS_INFO("y %f", config.y);
    ROS_INFO("z %f", config.z);
    if (config.z > 0.0f)
        mapper.epipolar(config.x, config.y, config.z);

    cv::Mat result = mapper.output();

    sendCloud(result, img1);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_mapper");
    ros::NodeHandle n("~");
    f_out = fopen("/home/yzf/outliers.txt", "w");

    ROS_INFO("read parameter");

    CALIB_DIR = readParam<std::string>(n, "calib_dir");
    CAM_NAME = readParam<std::string>(n, "cam_name");

    std::cout << CALIB_DIR + CAM_NAME + "/left.yml" << std::endl;
    cv::FileStorage param_reader_l(CALIB_DIR + CAM_NAME + "/left.yml", cv::FileStorage::READ);
    param_reader_l["camera_matrix"] >> K1;
    param_reader_l["distortion_coefficients"] >> D1;
/*
    param_reader_l["T_BS"] >> T_BS;
    cv::cv2eigen(T_BS.rowRange(0, 3).colRange(0, 3), R_bs);
    cv::cv2eigen(T_BS.rowRange(0, 3).colRange(3, 4), P_bs);
*/

    cv::cv2eigen(K1, K_eigen);
    cv::cv2eigen(K, K_eigen);
    std::cout << K_eigen << std::endl;
    mapper.initIntrinsic(K1, D1, K1, D1);

/*
    cv::FileStorage param_reader_r(CALIB_DIR + CAM_NAME + "/right.yml", cv::FileStorage::READ);
    param_reader_r["camera_matrix"] >> K2;
    param_reader_r["distortion_coefficients"] >> D2;

    cv::FileStorage param_reader_stereo(CALIB_DIR + CAM_NAME + "/stereo.yml", cv::FileStorage::READ);
    param_reader_stereo["stereo_r"] >> R21;
    param_reader_stereo["stereo_t"] >> T21;
*/

    std::cout << K1 << std::endl;
    std::cout << D1 << std::endl;
/*
    std::cout << K2 << std::endl;
    std::cout << D2 << std::endl;
    std::cout << R21 << std::endl;
    std::cout << T21 << std::endl;
*/

    for (int i = 1; i <= DEP_CNT; i++)
        std::cout << 1.0f / (DEP_SAMPLE * i) << std::endl;

/*
    cv::stereoRectify(K1, D1, K2, D2, cv::Size(WIDTH * 2, HEIGHT * 2), R21, T21, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 0);
    cv::initUndistortRectifyMap(K1, D1, R1, P1, cv::Size(WIDTH * 2, HEIGHT * 2), CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, cv::Size(WIDTH * 2, HEIGHT * 2), CV_16SC2, map21, map22);
*/

    //cv::namedWindow("motion stereo");
    //cv::namedWindow("spatial stereo");
    //cv::setMouseCallback("ep line", onMouse, NULL);

    message_filters::Subscriber<sensor_msgs::Image> sub_img_l(n, "image_l", 100);
    message_filters::Subscriber<sensor_msgs::Image> sub_img_r(n, "image_r", 100);
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_ref_pose(n, "ref_pose", 2);
    message_filters::Subscriber<geometry_msgs::PoseStamped> sub_cur_pose(n, "cur_pose", 2);

#if STEREO
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(1000), sub_img_l, sub_img_r, sub_pose);
    sync.registerCallback(boost::bind(&callback_img2, _1, _2, _3));
#else

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy_img;
    message_filters::Synchronizer<MySyncPolicy_img> sync_img(MySyncPolicy_img(1000), sub_img_l, sub_img_r);
    sync_img.registerCallback(boost::bind(&callback_raw_image, _1, _2));

    typedef message_filters::sync_policies::ExactTime<geometry_msgs::PoseStamped, geometry_msgs::PoseStamped> MySyncPolicy_pose;
    message_filters::Synchronizer<MySyncPolicy_pose> sync_pose(MySyncPolicy_pose(1000), sub_ref_pose, sub_cur_pose);
    sync_pose.registerCallback(boost::bind(&callback_raw_pose, _1, _2));
#endif

    pub_point_cloud2 = n.advertise<sensor_msgs::PointCloud2>("point_cloud2", 1000);
    pub_point_cloud_ref = n.advertise<sensor_msgs::PointCloud2>("point_cloud_ref", 1000);

    pub_color_img = n.advertise<sensor_msgs::Image>("rgb/image_raw", 1000);
    pub_color_img_info = n.advertise<sensor_msgs::CameraInfo>("rgb/image_info", 1000);

    pub_disp_img = n.advertise<sensor_msgs::Image>("depth/image_raw", 1000);
    pub_disp_img_info = n.advertise<sensor_msgs::CameraInfo>("depth/image_info", 1000);

    dynamic_reconfigure::Server<stereo_mapper::debugConfig> server;
    dynamic_reconfigure::Server<stereo_mapper::debugConfig>::CallbackType f;

    f = boost::bind(&callback_debug, _1, _2);
    server.setCallback(f);

    ros::spin();
    return 0;
}
