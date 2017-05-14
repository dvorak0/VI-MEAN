#include "stereo_mapper.h"

StereoMapper::StereoMapper()
    : raw_cost(1, HEIGHT * ALIGN_WIDTH * DEP_CNT, CV_32F),
      sgm_cost(1, HEIGHT * ALIGN_WIDTH * DEP_CNT, CV_32F),
      dep(HEIGHT, WIDTH, CV_32F),
      tmp(1, WIDTH * DEP_CNT, CV_32F)
{
}

void StereoMapper::initIntrinsic(const cv::Mat &_K1, const cv::Mat &_D1, const cv::Mat &_K2, const cv::Mat &_D2)
{

    nK1 = _K1.clone();
    nK2 = _K2.clone();

#if DOWNSAMPLE
    nK1 /= 2;
    nK1.at<double>(2, 2) = 1;
    nK2 /= 2;
    nK2.at<double>(2, 2) = 1;
#endif

    cv::initUndistortRectifyMap(_K1, _D1, cv::Mat_<double>::eye(3, 3), nK1, cv::Size(WIDTH, HEIGHT), CV_32FC1, _map1_l, _map2_l);
    map1_l.upload(_map1_l);
    map2_l.upload(_map2_l);

    cv::initUndistortRectifyMap(_K2, _D2, cv::Mat_<double>::eye(3, 3), nK2, cv::Size(WIDTH, HEIGHT), CV_32FC1, _map1_r, _map2_r);
    map1_r.upload(_map1_r);
    map2_r.upload(_map2_r);
}

void StereoMapper::initReference(const cv::Mat &_img_l)
{
    ROS_INFO("init");

    cv::Mat tmp_img;
    _img_l.convertTo(tmp_img, CV_32F);
    raw_img_l.upload(tmp_img);

    cv::gpu::remap(raw_img_l, img_l, map1_l, map2_l, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    img_l.download(tmp_img);
    tmp_img.convertTo(img_intensity, CV_8U);

    //cv::imwrite("/home/ubuntu/left.jpg", img_intensity);
    //cv::imwrite("/home/ubuntu/raw_left.jpg", _img_l);

    measurement_cnt = 0;
}

void StereoMapper::update(const cv::Mat &_img_r, const cv::Mat &R_l, const cv::Mat &T_l, const cv::Mat &R_r, const cv::Mat &T_r)
{
    measurement_cnt++;
    ROS_INFO("update %d", measurement_cnt);
    double t = cv::getTickCount();

    cv::Mat tmp_img;
    _img_r.convertTo(tmp_img, CV_32F);
    raw_img_r.upload(tmp_img);

    cv::gpu::remap(raw_img_r, img_r, map1_r, map2_r, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

#if BENCHMARK
    img_r.download(tmp_img);
    tmp_img.convertTo(img_intensity_r, CV_8U);
#endif

    //img_l.download(tmp_img);
    //tmp_img.convertTo(img_intensity_r, CV_8U);
    //cv::imwrite("/home/ubuntu/right.jpg", img_intensity_r);
    //cv::imwrite("/home/ubuntu/raw_right.jpg", _img_r);

    printf("PW CUDA Time part 1: %fms\n", (cv::getTickCount() - t) / cv::getTickFrequency() * 1000);

    t = cv::getTickCount();

    R = nK2 * R_r.t() * R_l * nK1.inv();
    T = nK2 * R_r.t() * (T_l - T_r);
    ad_calc_cost(
        measurement_cnt,
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
        T.at<double>(0, 0), T.at<double>(0, 1), T.at<double>(0, 2),
        nK1.at<double>(0, 0), nK1.at<double>(1, 1), nK1.at<double>(0, 2), nK1.at<double>(1, 2),
        img_l.data, img_l.step,
        img_r.data, img_r.step,
        raw_cost.data);

    printf("PW CUDA Time part 2: %fms\n", (cv::getTickCount() - t) / cv::getTickFrequency() * 1000);
}

cv::Mat StereoMapper::output()
{
    int64 t = cv::getTickCount();

    sgm_cost.setTo(cv::Scalar_<float>(0.0));

    sgm2(img_l.data, img_l.step,
         img_l.data, img_l.step,
         raw_cost.data,
         sgm_cost.data);

    printf("PW CUDA Time part 3: %fms\n", (cv::getTickCount() - t) * 1000 / cv::getTickFrequency());

    t = cv::getTickCount();
    filter_cost(sgm_cost.data,
                dep.data, dep.step);
    printf("PW CUDA Time part 4: %fms\n", (cv::getTickCount() - t) * 1000 / cv::getTickFrequency());

    t = cv::getTickCount();
    cv::Mat result;
    dep.download(result);
    printf("PW CUDA Time part 5: %fms\n", (cv::getTickCount() - t) * 1000 / cv::getTickFrequency());
    return result;
}

void StereoMapper::epipolar(double x, double y, double z)
{
    cv::Mat P{3, 1, CV_64F};
    P.at<double>(0, 0) = x;
    P.at<double>(1, 0) = y;
    P.at<double>(2, 0) = z;
    cv::Mat p = nK1 * P;
    double uu = p.at<double>(0, 0) / p.at<double>(2, 0);
    double vv = p.at<double>(1, 0) / p.at<double>(2, 0);
    int u = uu + 0.5;
    int v = vv + 0.5;
    printf("%f %f %d %d\n", uu, vv, u, v);
    cv::Mat img;
    img_r.download(img);
    cv::Mat pp{3, 1, CV_64F};
    pp.at<double>(0, 0) = u;
    pp.at<double>(1, 0) = v;
    pp.at<double>(2, 0) = 1;

    cv::Mat imgl = img_intensity.clone();
    cv::circle(imgl, cv::Point(u, v), 2, cv::Scalar(-1));

    for (int i = 0; i < DEP_CNT; i++)
    {
        cv::Mat ppp = R * pp;
        double idep = i * DEP_SAMPLE;
        double x = ppp.at<double>(0, 0);
        double y = ppp.at<double>(1, 0);
        double z = ppp.at<double>(2, 0);

        float w = z + T.at<double>(2, 0) * idep;
        float u = (x + T.at<double>(0, 0) * idep) / w;
        float v = (y + T.at<double>(1, 0) * idep) / w;
        printf("%f %f\n", u, v);
        cv::circle(img, cv::Point(u, v), 2, cv::Scalar(-1));
    }
    cv::imshow("r", imgl);
    cv::imshow("m", img);
    cv::waitKey(10);
}
