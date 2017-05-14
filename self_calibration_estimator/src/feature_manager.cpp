#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
    ROS_INFO("init feature_manager");
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i] = _ric[i];
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    return traversal().size();
}

bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Vector3d>>> &image, const map<int, Vector3d> &debug_image)
{
    ROS_INFO("input feature: %lu", image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    for (auto &id_pts : image)
    {
        vector<FeaturePerCamera> f_per_cam;
        for (auto &i_p : id_pts.second)
        {
            if (i_p.second.z() > 0 && i_p.second.head<2>().norm() < 2)
                f_per_cam.push_back(FeaturePerCamera(i_p.first, i_p.second));
            //printf("%d (%f,%f) ", i_p.first, i_p.second.x(), i_p.second.y());
        }
        if (f_per_cam.size() == 0)
            continue;
        //puts("");
        FeaturePerFrame f_per_fra(move(f_per_cam));

        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it) {
            return it.feature_id == feature_id;
        });

        if (it == feature.end())
        {
            feature.push_back(FeaturePerId(feature_id, frame_count));

            feature.back().feature_per_frame.push_back(f_per_fra);
#ifdef GT
            feature.back().gt_p = debug_image.at(id_pts.first);
#endif

            //if (frame_count < WINDOW_SIZE)
            //    feature.back().feature_per_frame.back().is_used = compensatedParallax1(feature.back()) > MIN_PARALLAX_POINT / 100;
            //else
            feature.back().feature_per_frame.back().is_used = compensatedParallax1(feature.back()) > MIN_PARALLAX_POINT;

            feature.back().used_num += feature.back().feature_per_frame.back().is_used;
        }
        else if (it->feature_id == feature_id)
        {
            it->feature_per_frame.push_back(f_per_fra);

            //ROS_DEBUG("1 %d, %lu, %d, %f", it->start_frame, it->feature_per_frame.size(), it->used_num, FOCAL_LENGTH * compensatedParallax1(*it));
            //if (frame_count < WINDOW_SIZE)
            //    it->feature_per_frame.back().is_used = compensatedParallax1(*it) > MIN_PARALLAX_POINT / 100;
            //else
            //double parallax = compensatedParallax1(*it);
            //it->feature_per_frame.back().parallax = parallax;
            it->feature_per_frame.back().is_used = compensatedParallax1(*it) > MIN_PARALLAX_POINT;
            it->used_num += it->feature_per_frame.back().is_used;

            if (it->used_num == 1 && it->feature_per_frame.back().is_used)
            {
                feature.splice(feature.end(), feature, it);
            }
        }
    }

    if (frame_count < WINDOW_SIZE)
        return true;

    for (auto &it_per_id : feature)
    {
        if (it_per_id.start_frame <= WINDOW_SIZE - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= WINDOW_SIZE - 1)
        {
            parallax_sum += compensatedParallax2(it_per_id);
            parallax_num++;
        }
    }

    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_INFO("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_INFO("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        printf("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            printf("%d,", int(j.is_used));
            sum += j.is_used;
            for (auto &k : j.feature_per_camera)
                printf("%d,(%lf,%lf) ", k.camera_id, k.point(0), k.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
        puts("\n");
    }
}

void FeatureManager::htmlShow()
{
    FILE *file = fopen("/home/dvorak/debug.html", "w");
    fprintf(file, "<html>\n");
    fprintf(file, "<head>\n");
    fprintf(file, "<title>Feature Manager Debug</title>\n");
    fprintf(file, "<script type='text/javascript' src='https://ajax.googleapis.com/ajax/libs/jquery/1.12.0/jquery.min.js'></script>\n");
    fprintf(file, "<link rel='stylesheet' type='text/css' href='mystyle.css'>");
    fprintf(file, "</header>\n");
    fprintf(file, "<body>\n");
    fprintf(file, "<table border='1'>\n");
    fprintf(file, "<tr><td>ID</td><td>used_num</td>");
    for (int i = 0; i <= WINDOW_SIZE; i++)
        fprintf(file, "<td>%d</td>", i);
    fprintf(file, "</tr>\n");

    for (const auto &f_per_id : feature)
    {
        for (int k = 0; k < NUM_OF_CAM; k++)
        {
            if (f_per_id.used_num == 0)
                fprintf(file, "<tr class='unused'>\n");
            else
                fprintf(file, "<tr>\n");

            if (k == 0)
            {
                fprintf(file, "<td rowspan='%d'>%d</td>\n", NUM_OF_CAM, f_per_id.feature_id);
                fprintf(file, "<td rowspan='%d'>%d</td>\n", NUM_OF_CAM, f_per_id.used_num);
            }

            int i = 0;
            for (; i < f_per_id.start_frame; i++)
                fprintf(file, "<td></td>\n");
            //fprintf(file, "<td bgcolor='#0000FF'>0.0</td>\n");
            //i++;
            for (int j = 0; j < static_cast<int>(f_per_id.feature_per_frame.size()); j++, i++)
            {
                if (k < static_cast<int>(f_per_id.feature_per_frame[j].feature_per_camera.size()))
                {
                    if (f_per_id.feature_per_frame[j].is_used)
                        fprintf(file, "<td bgcolor='#00FF00'>%f</td>\n", f_per_id.feature_per_frame[j].feature_per_camera[k].parallax * FOCAL_LENGTH);
                    else
                        fprintf(file, "<td bgcolor='#FF0000'>%f</td>\n", f_per_id.feature_per_frame[j].feature_per_camera[k].parallax * FOCAL_LENGTH);
                }
                else
                    fprintf(file, "<td>NULL</td>\n");
            }
            for (; i <= WINDOW_SIZE; i++)
                fprintf(file, "<td></td>\n");

            fprintf(file, "</tr>\n");
        }
    }

    fprintf(file, "</table>\n");
    fprintf(file, "</body>\n");
    fprintf(file, "</html>\n");
    fclose(file);
}

void FeatureManager::shift(int n_start_frame)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->endFrame() >= n_start_frame)
        {
            vector<FeaturePerFrame> n_feature_points;
            n_feature_points.swap(it->feature_per_frame);
            it->used_num = 0;

            int o_start_frame = it->start_frame;
            it->start_frame = max(0, o_start_frame - n_start_frame);

            for (int i = max(0, n_start_frame - o_start_frame); i < int(n_feature_points.size()); i++)
            {
                it->feature_per_frame.push_back(FeaturePerFrame(move(n_feature_points[i].feature_per_camera)));
                it->feature_per_frame.back().is_used = compensatedParallax1(*it) > MIN_PARALLAX_POINT;
                if (it->feature_per_frame.back().is_used)
                    it->used_num++;
            }
        }
        else
        {
            feature.erase(it);
        }
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count, int camera_id)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature)
    {
        int l = int(it.feature_per_frame.size());
        if (l >= 2 && it.start_frame + l - 1 == frame_count)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            bool has_a = false, has_b = false;
            for (unsigned int i = 0; i < NUM_OF_CAM; i++)
            {
                if (i < it.feature_per_frame[l - 2].feature_per_camera.size() &&
                    it.feature_per_frame[l - 2].feature_per_camera[i].camera_id == camera_id)
                {
                    has_a = true;
                    a = it.feature_per_frame[l - 2].feature_per_camera[i].point;
                }

                if (i < it.feature_per_frame[l - 1].feature_per_camera.size() &&
                    it.feature_per_frame[l - 1].feature_per_camera[i].camera_id == camera_id)
                {
                    has_b = true;
                    b = it.feature_per_frame[l - 1].feature_per_camera[i].point;
                }
            }
            if (has_a && has_b)
                corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}

vector<FeaturePerId *> FeatureManager::traversal()
{
    vector<FeaturePerId *> ans;
    int sum = 0;
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);
        ROS_ASSERT(it.start_frame + int(it.feature_per_frame.size()) - 1 <= WINDOW_SIZE);

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2)
        {
            sum++;
            ans.push_back(&it);
        }
    }
    return ans;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto it_per_id : traversal())
        it_per_id->estimated_depth = 1.0 / x(++feature_index);
}

//void FeatureManager::updateDepth(const VectorXd &x)
//{
//    int feature_index = -1;
//    for (auto it_per_id : traversal())
//    {
//#ifdef INV_DEP
//        double x1 = 1. / it_per_id->estimated_depth;
//        x1 += x(++feature_index);
//        it_per_id->estimated_depth = 1. / x1;
//#else
//        it_per_id->estimated_depth += x(++feature_index);
//#endif
//    }
//}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (auto it_per_id : traversal())
#if 1
        dep_vec(++feature_index) = 1. / it_per_id->estimated_depth;
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    outlier_info.clear();
    for (auto it_per_id : traversal())
    {
        if (it_per_id->estimated_depth > 0)
            continue;
        int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
        int camera_i = it_per_id->feature_per_frame[0].feature_per_camera[0].camera_id, camera_j;
        Vector3d p_i = Ps[imu_i], p_j;
        Vector3d pts_i = it_per_id->feature_per_frame[0].feature_per_camera[0].point, pts_j;

        double A = 0, b = 0;
        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id->feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id->feature_per_frame)
        {
            imu_j++;
            for (auto &it_per_camera : it_per_frame.feature_per_camera)
            {
                camera_j = it_per_camera.camera_id;

                Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
                Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
                Eigen::Vector3d t = R0.transpose() * (t1 - t0);
                Eigen::Matrix3d R = R0.transpose() * R1;
                Eigen::Matrix<double, 3, 4> P;
                P.leftCols<3>() = R.transpose();
                P.rightCols<1>() = -R.transpose() * t;
                Eigen::Vector3d f = it_per_camera.point.normalized();
                svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
                svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

                if (imu_i == imu_j && camera_i == camera_j)
                    continue;
                p_j = Ps[imu_j];
                pts_j = it_per_camera.point;

                MatrixXd reduce(2, 3);
                reduce << 1, 0, -pts_j(0),
                    0, 1, -pts_j(1);
                MatrixXd tmp_A = reduce * ric[camera_j].inverse() * Rs[imu_j].inverse() * Rs[imu_i] * ric[camera_i] * pts_i;
                Vector2d tmp_b = reduce * ric[camera_j].inverse() * (Rs[imu_j].inverse() * Rs[imu_i] * tic[camera_i] + Rs[imu_j].inverse() * (p_i - p_j) - tic[camera_j]);
                A += (tmp_A.transpose() * tmp_A)(0, 0);
                b += (tmp_A.transpose() * tmp_b)(0, 0);
            }
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();

        double my_method = -b / A;
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id->estimated_depth = svd_method;
        it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id->estimated_depth < 0.1)
        {
            std::vector<int> window_id;
            it_per_id->estimated_depth = INIT_DEPTH;

#if 0
            int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
            for (auto &it_per_frame : it_per_id->feature_per_frame)
            {
                imu_j++;
                for (auto &it_per_camera : it_per_frame.feature_per_camera)
                {
                    camera_j = it_per_camera.camera_id;
                    p_j = Ps[imu_j];
                    pts_j = Rs[imu_j] * ric[0] * it_per_camera.point;
                    window_id.push_back(imu_j);
                    ROS_INFO("%d,%d,(%f,%f,%f)(%f,%f,%f) ", it_per_id->feature_id, imu_j, p_j.x(), p_j.y(), p_j.z(), pts_j.x(), pts_j.y(), pts_j.z());
                    const int K = 100;
                    const double MIN_DIST = 0.1;
                    const double DEP_SAMPLE = (1.0 / MIN_DIST) / K;
                    for (int k = 0; k < K; k++)
                    {
                        double dep = 0;
                        if (k == 0)
                            dep = 1e3;
                        else
                            dep = 1. / (k * DEP_SAMPLE);

                        pts_j = ric[0].transpose() * (Rs[imu_i].transpose() * (Rs[imu_j] * (ric[0] * pts_i * dep + tic[0]) + Ps[imu_j] - Ps[imu_i]) - tic[0]);
                        pts_j /= pts_j.z();
                        double cost = (pts_j.head<2>() - it_per_camera.point.head<2>()).norm();
                        printf("(%f %f) ", dep, cost);
                    }
                    puts("");
                }
            }
#endif
            outlier_info.emplace_back(it_per_id->feature_id, window_id);
        }

        ROS_INFO("triangulation feature_id: %d, estimated_depth: (%f,%f), (%f,%f)", it_per_id->feature_id,
                 my_method, svd_method,
                 A, svd_V[3]);
        continue;

        for (int i = 0; i < 3; i++)
        {
            A = 0;
            b = 0;

            imu_j = imu_i - 1;

            double dep_i = it_per_id->estimated_depth, dep_j;
            Vector3d pts_camera_i = pts_i * dep_i, pts_camera_j;
            Vector3d pts_imu_i = ric[camera_i] * pts_camera_i + tic[camera_i];
            Vector3d pts_world = Rs[imu_i] * pts_imu_i + p_i;

            for (auto &it_per_frame : it_per_id->feature_per_frame)
            {
                imu_j++;
                for (auto &it_per_camera : it_per_frame.feature_per_camera)
                {
                    camera_j = it_per_camera.camera_id;
                    if (imu_i == imu_j && camera_i == camera_j)
                        continue;
                    p_j = Ps[imu_j];
                    pts_j = it_per_camera.point;

                    Vector3d pts_imu_j = Rs[imu_j].transpose() * (pts_world - p_j);
                    Vector3d pts_camera_j = ric[camera_j].transpose() * (pts_imu_j - tic[camera_j]);
                    dep_j = pts_camera_j(2);

                    Matrix<double, 2, 3> reduce(2, 3);
                    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

                    Vector2d tmp_b = Vector2d(pts_camera_j(0) / dep_j - pts_j(0),
                                              pts_camera_j(1) / dep_j - pts_j(1));

                    Vector2d tmp_A = reduce * Rs[imu_j].transpose() * Rs[imu_i] * ric[camera_i] * pts_i * -(dep_i * dep_i);

                    A += tmp_A.transpose() * tmp_A;
                    b += tmp_A.transpose() * tmp_b;
                }
            }
            double dx = -b / A;
            it_per_id->estimated_depth = 1 / (1 / dep_i + dx);
        }
        ROS_INFO("2 triangulation feature_id: %d, estimated_depth: %f, b: %f, A: %f", it_per_id->feature_id, it_per_id->estimated_depth, b, A);
    }
}

void FeatureManager::tagMarginalizedPoints(bool is_non_linear, bool marginalization_flag)
{
    //vector<FeaturePerId *> sorted_feature;

    //int whole_num = 0;
    for (auto it_per_id : traversal())
    {
        it_per_id->is_margin = false;
        int imu_i = it_per_id->start_frame, imu_j = imu_i - 1;
        int camera_i = it_per_id->feature_per_frame[0].feature_per_camera[0].camera_id, camera_j;
        double sum_dep_gradient = 0.0;
        int proj_cnt = 0;
        for (auto &it_per_frame : it_per_id->feature_per_frame)
        {
            imu_j++;
            for (auto &it_per_camera : it_per_frame.feature_per_camera)
            {
                camera_j = it_per_camera.camera_id;
                if (imu_i == imu_j && camera_i == camera_j)
                    continue;
                sum_dep_gradient += it_per_camera.dep_gradient;
                proj_cnt++;
            }
        }

        if (is_non_linear)
        {
            //if (sum_dep_gradient < 1e-5)
            //{
            //    printf("%d %d %d: %f\n", it_per_id->used_num, imu_i, imu_j, sum_dep_gradient);
            //}
            //ROS_ASSERT(sum_dep_gradient >= 1e-5);
        }

        if (!marginalization_flag && it_per_id->start_frame == 0)
            it_per_id->is_margin = true;

        if (marginalization_flag)
        {
            if (it_per_id->start_frame == WINDOW_SIZE - 1)
                it_per_id->is_margin = true;
            else if (it_per_id->start_frame < WINDOW_SIZE - 1 && it_per_id->endFrame() >= WINDOW_SIZE - 1 && it_per_id->used_num - it_per_id->feature_per_frame[WINDOW_SIZE - 1 - it_per_id->start_frame].is_used == 0)
                it_per_id->is_margin = true;
        }

        //if (!it_per_id->is_margin)
        //{
        //    sorted_feature.push_back(it_per_id);
        //    whole_num++;
        //}
    }

    //std::stable_sort(sorted_feature.begin(), sorted_feature.end(),
    //                 [](FeaturePerId *a, FeaturePerId *b)
    //                 {
    //        return a->endFrame()<b->endFrame();
    //                 });

    //int remove_count = max(0, whole_num - MAX_FEATURE_CNT);
    //for (auto it : sorted_feature)
    //{
    //    if (remove_count)
    //    {
    //        it->is_margin = true;
    //        remove_count--;
    //    }
    //    else
    //        break;
    //}
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBack(int frame_count, int n_calibration, int n_state, vector<int> &remove, vector<int> &not_remove)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count, int n_calibration, int n_state, vector<int> &remove, vector<int> &not_remove)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax1(FeaturePerId &f_per_id)
{
    int l = f_per_id.feature_per_frame.size();
    FeaturePerFrame &frame_i = f_per_id.feature_per_frame[0];
    FeaturePerFrame &frame_j = f_per_id.feature_per_frame[l - 1];

    int r_i = f_per_id.start_frame + 0;
    int r_j = f_per_id.start_frame + l - 1;

    int camera_id_i = frame_i.feature_per_camera[0].camera_id;
    Vector3d p_i = frame_i.feature_per_camera[0].point;

    double u_i = p_i(0);
    double v_i = p_i(1);

    //puts("");
    double ans = 0;
    for (int k = 0; k < int(frame_j.feature_per_camera.size()); k++)
    {
        int camera_id_j = frame_j.feature_per_camera[k].camera_id;
        Vector3d p_j = frame_j.feature_per_camera[k].point;
        if (COMPENSATE_ROTATION)
            p_j = Quaterniond(ric[camera_id_i].transpose() * Rs[r_i].transpose() * Rs[r_j] * ric[camera_id_j]).normalized() * p_j;

        double dep_j = p_j(2);
        double u_j = p_j(0) / dep_j;
        double v_j = p_j(1) / dep_j;

        double du = u_i - u_j, dv = v_i - v_j;
        double para = sqrt(du * du + dv * dv);

        frame_j.feature_per_camera[k].parallax = para;

        if (r_i == r_j && camera_id_i == camera_id_j && l == 1 && k == 0)
        {
            if (!(para < 1e-3))
            {
                cout << frame_i.feature_per_camera[0].point << endl;
                cout << frame_j.feature_per_camera[k].point << endl;
                cout << p_j.transpose() << endl;
                cout << ric[camera_id_i].transpose() * Rs[r_i].transpose() * Rs[r_j] * ric[camera_id_j] << endl;
            }
            ROS_ASSERT_MSG(para < 1e-2, "id: %d, parallax: %f, used_num: %d, length: %lu, start_frame: %d",
                           f_per_id.feature_id,
                           para,
                           f_per_id.used_num,
                           f_per_id.feature_per_frame.size(),
                           f_per_id.start_frame);
        }

        //double aa = u_i * u_j + v_i * v_j + 1.;
        //double la = sqrt(u_i * u_i + v_i * v_i + 1.);
        //double lb = sqrt(u_j * u_j + v_j * v_j + 1.);
        //double r_angle = acos(aa / (la * lb)) / M_PI * 180;

        //if (r_angle < 5.0)
        //    para = 0.0;
        //if (camera_id_i != camera_id_j)
        //    para *= 2;

        ans = max(ans, para);
    }

    return ans;
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id)
{
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[WINDOW_SIZE - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[WINDOW_SIZE - 1 - it_per_id.start_frame];

    int r_i = WINDOW_SIZE - 2;
    int r_j = WINDOW_SIZE - 1;

    double ans = 0;
    for (int k = 0; k < int(frame_j.feature_per_camera.size()); k++)
    {
        int camera_id_j = frame_j.feature_per_camera[k].camera_id;
        Vector3d p_j = frame_j.feature_per_camera[k].point;

        double u_j = p_j(0);
        double v_j = p_j(1);

        auto it = std::find_if(frame_i.feature_per_camera.begin(),
                               frame_i.feature_per_camera.end(),
                               [camera_id_j](const FeaturePerCamera &f_per_cam) {
                                   return camera_id_j == f_per_cam.camera_id;
                               });

        if (it == frame_i.feature_per_camera.end())
            continue;

        int camera_id_i = it->camera_id;
        Vector3d p_i = it->point;
        if (COMPENSATE_ROTATION)
            p_i = Quaterniond(ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i]).normalized() * p_i;

        double dep_i = p_i(2);
        double u_i = p_i(0) / dep_i;
        double v_i = p_i(1) / dep_i;

        double du = u_i - u_j, dv = v_i - v_j;
        ans = max(ans, sqrt(du * du + dv * dv));
    }

    return ans;
}
