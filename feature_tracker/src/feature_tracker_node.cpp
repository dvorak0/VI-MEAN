#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <mutex>
#include <condition_variable>
#include <thread>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;

std::mutex m_buf;
std::condition_variable con;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img;

FeatureTracker trackerData[NUM_OF_CAM];

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();
}

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img_buf.push(img_msg);
    m_buf.unlock();
    con.notify_one();
}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::ImageConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::ImageConstPtr>> measurements;

    while (true)
    {
        if (imu_buf.empty() || img_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp > img_buf.front()->header.stamp))
        {
            ROS_WARN("wait for imu, only should happen at the beginning");
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp < img_buf.front()->header.stamp))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            img_buf.pop();
            continue;
        }
        sensor_msgs::ImageConstPtr img_msg = img_buf.front();
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

void process()
{
    puts("begin");
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::ImageConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });

        ROS_INFO("got measurement of size: %lu", measurements.size());

        lk.unlock();

        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;

            ROS_INFO("begin: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());

            ROS_INFO("processing vision data with stamp %f", img_msg->header.stamp.toSec());

            cv_bridge::CvImageConstPtr ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

            TicToc t_r;
            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                ROS_DEBUG("processing camera %d", i);
                if (i != 1 || !STEREO_TRACK)
                    trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)));
                else
                {
                    if (EQUALIZE)
                    {
                        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                        clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
                    }
                    else
                        trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
                }

#if SHOW_UNDISTORTION
                trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
            }

            if (FeatureTracker::img_cnt == 0 && STEREO_TRACK && trackerData[0].cur_pts.size() > 0)
            {
                r_status.clear();
                r_err.clear();
                TicToc t_o;
                cv::calcOpticalFlowPyrLK(trackerData[0].cur_img, trackerData[1].cur_img, trackerData[0].cur_pts, trackerData[1].cur_pts, r_status, r_err, cv::Size(21, 21), 3);
                ROS_DEBUG("spatial optical flow costs: %fms", t_o.toc());
                vector<cv::Point2f> ll, rr;
                vector<int> idx;
                for (unsigned int i = 0; i < r_status.size(); i++)
                {
                    if (!inBorder(trackerData[1].cur_pts[i]))
                        r_status[i] = 0;

                    if (r_status[i])
                    {
                        idx.push_back(i);

                        Eigen::Vector3d tmp_p;
                        trackerData[0].m_camera->liftProjective(Eigen::Vector2d(trackerData[0].cur_pts[i].x, trackerData[0].cur_pts[i].y), tmp_p);
                        tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                        tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                        ll.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));

                        trackerData[1].m_camera->liftProjective(Eigen::Vector2d(trackerData[1].cur_pts[i].x, trackerData[1].cur_pts[i].y), tmp_p);
                        tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                        tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                        rr.push_back(cv::Point2f(tmp_p.x(), tmp_p.y()));
                    }
                }
                if (ll.size() >= 8)
                {
                    vector<uchar> status;
                    TicToc t_f;
                    cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 1.0, 0.5, status);
                    ROS_DEBUG("find f cost: %f", t_f.toc());
                    int r_cnt = 0;
                    for (unsigned int i = 0; i < status.size(); i++)
                    {
                        if (status[i] == 0)
                            r_status[idx[i]] = 0;
                        r_cnt += r_status[idx[i]];
                    }
                }
            }

            for (unsigned int i = 0;; i++)
            {
                bool completed = false;
                for (int j = 0; j < NUM_OF_CAM; j++)
                    if (j != 1 || !STEREO_TRACK)
                        completed |= trackerData[j].updateID(i);
                if (!completed)
                    break;
            }

            if (FeatureTracker::img_cnt == 0)
            {
                sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
                sensor_msgs::ChannelFloat32 id_of_point;
                sensor_msgs::ChannelFloat32 u_of_point;
                sensor_msgs::ChannelFloat32 v_of_point;

                feature_points->header = img_msg->header;
                feature_points->header.frame_id = "world";

                vector<set<int>> hash_ids(NUM_OF_CAM);
                for (int i = 0; i < NUM_OF_CAM; i++)
                {
                    if (i != 1 || !STEREO_TRACK)
                    {
                        auto un_pts = trackerData[i].undistortedPoints();
                        auto &cur_pts = trackerData[i].cur_pts;
                        auto &ids = trackerData[i].ids;
                        for (unsigned int j = 0; j < ids.size(); j++)
                        {
                            int p_id = ids[j];
                            hash_ids[i].insert(p_id);
                            geometry_msgs::Point32 p;
                            p.x = un_pts[j].x;
                            p.y = un_pts[j].y;
                            p.z = 1;

                            feature_points->points.push_back(p);
                            id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                            u_of_point.values.push_back(cur_pts[j].x);
                            v_of_point.values.push_back(cur_pts[j].y);
                            ROS_ASSERT(inBorder(cur_pts[j]));
                        }
                    }
                    else if (STEREO_TRACK)
                    {
                        auto r_un_pts = trackerData[1].undistortedPoints();
                        auto &ids = trackerData[0].ids;
                        for (unsigned int j = 0; j < ids.size(); j++)
                        {
                            if (r_status[j])
                            {
                                int p_id = ids[j];
                                hash_ids[i].insert(p_id);
                                geometry_msgs::Point32 p;
                                p.x = r_un_pts[j].x;
                                p.y = r_un_pts[j].y;
                                p.z = 1;

                                feature_points->points.push_back(p);
                                id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                            }
                        }
                    }
                }
                feature_points->channels.push_back(id_of_point);
                feature_points->channels.push_back(u_of_point);
                feature_points->channels.push_back(v_of_point);
                ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
                pub_img.publish(feature_points);

                if (SHOW_TRACK)
                {
                    //ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);

                    cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
                    //cv::Mat stereo_img = ptr->image;

                    for (int i = 0; i < NUM_OF_CAM; i++)
                    {
                        cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                        cv::cvtColor(trackerData[i].cur_img, tmp_img, CV_GRAY2RGB);
                        if (i != 1 || !STEREO_TRACK)
                        {
                            for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                            {
                                double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                                cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                                char name[10];
                                sprintf(name, "%d", trackerData[i].ids[j]);
                                cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                            }
                        }
                        else
                        {
                            for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                            {
                                if (r_status[j])
                                {
                                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(0, 255, 0), 2);
                                    cv::line(stereo_img, trackerData[i - 1].cur_pts[j], trackerData[i].cur_pts[j] + cv::Point2f(0, ROW), cv::Scalar(0, 255, 0));
                                }
                            }
                        }
                    }
                    cv::imshow("vis", stereo_img);
                    cv::waitKey(5);
                    //pub_match.publish(ptr->toImageMsg());
                }
            }

            FeatureTracker::img_cnt = (FeatureTracker::img_cnt + 1) % FREQ;
            ROS_DEBUG("whole processing costs: %f\n\n\n", t_r.toc());
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug);
    readParameters(n);

    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CALIB_DIR + CAM_NAMES[i] + string("_camera_calib.yaml"));

    ros::Subscriber sub_imu = n.subscribe("imu", 1000, imu_callback);
    ros::Subscriber sub_img = n.subscribe("raw_image", 100, img_callback);

    pub_img = n.advertise<sensor_msgs::PointCloud>("image", 1000);

    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);

    std::thread loop{process};
    ros::spin();
    return 0;
}
