
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#define _USE_MATH_DEFINES
#include <math.h>
#include <filesystem>
#include <libexif/exif-data.h>
#include <libexif/exif-loader.h>

#include <numeric>

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#include "euler_order.h"
#include "quaternion.h"

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.3;
double seam_megapix = 0.1;
double compose_megapix = 0.4;
float conf_thresh = 1.f;
#ifdef HAVE_OPENCV_XFEATURES2D
string features_type = "surf";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.6f;
#endif
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "reproj";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 64;
string seam_find_type = "dp_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;

template<typename TFloat>
cv::Vec<TFloat, 3> rotationMatrixToEulerAngles(Mat R, EulerOrder order)
{
    TFloat x, y, z;
    auto m11 = R.at<TFloat>(0, 0);
    auto m12 = R.at<TFloat>(0, 1);
    auto m13 = R.at<TFloat>(0, 2);
    auto m21 = R.at<TFloat>(1, 0);
    auto m22 = R.at<TFloat>(1, 1);
    auto m23 = R.at<TFloat>(1, 2);
    auto m31 = R.at<TFloat>(2, 0);
    auto m32 = R.at<TFloat>(2, 1);
    auto m33 = R.at<TFloat>(2, 2);

    switch (order) {

    case EulerOrder::XYZ:

        y = std::asin(std::clamp(m13, TFloat(-1), TFloat(1)));

        if (std::abs(m13) < 0.9999999) {
            x = std::atan2(-m23, m33);
            z = std::atan2(-m12, m11);
        }
        else {
            x = std::atan2(m32, m22);
            z = 0;
        }

        break;

    case EulerOrder::YXZ:

        x = std::asin(-std::clamp(m23, TFloat(-1), TFloat(1)));

        if (std::abs(m23) < 0.9999999) {

            y = std::atan2(m13, m33);
            z = std::atan2(m21, m22);

        }
        else {

            y = std::atan2(-m31, m11);
            z = 0;

        }

        break;

    case EulerOrder::ZXY:

        x = std::asin(std::clamp(m32, TFloat(-1), TFloat(1)));

        if (std::abs(m32) < 0.9999999) {

            y = std::atan2(-m31, m33);
            z = std::atan2(-m12, m22);

        }
        else {

            y = 0;
            z = std::atan2(m21, m11);

        }

        break;

    case EulerOrder::ZYX:

        y = std::asin(-std::clamp(m31, TFloat(-1), TFloat(1)));

        if (std::abs(m31) < 0.9999999) {

            x = std::atan2(m32, m33);
            z = std::atan2(m21, m11);

        }
        else {

            x = 0;
            z = std::atan2(-m12, m22);

        }

        break;

    case EulerOrder::YZX:

        z = std::asin(std::clamp(m21, TFloat(-1), TFloat(1)));

        if (std::abs(m21) < 0.9999999) {

            x = std::atan2(-m23, m22);
            y = std::atan2(-m31, m11);

        }
        else {

            x = 0;
            y = std::atan2(m13, m33);

        }

        break;

    case EulerOrder::XZY:

        z = std::asin(-std::clamp(m12, TFloat(-1), TFloat(1)));

        if (std::abs(m12) < 0.9999999) {

            x = std::atan2(m32, m22);
            y = std::atan2(m13, m11);

        }
        else {

            x = std::atan2(-m23, m33);
            y = 0;

        }

        break;
    default:
        assert(false);
    }
    return cv::Vec<TFloat, 3>(x, y, z);
}

template<typename TFloat>
cv::Mat eulerAnglesToRotationMatrix(cv::Vec<TFloat, 3>& euler, EulerOrder order)
{
    cv::Mat_<TFloat> ret(3, 3);
    TFloat te[16];

    auto x = euler[0];
    auto y = euler[1];
    auto z = euler[2];

    auto a = std::cos(x);
    auto b = std::sin(x);
    auto c = std::cos(y);
    auto d = std::sin(y);
    auto e = std::cos(z);
    auto f = std::sin(z);

    switch (order)
    {
    case EulerOrder::XYZ:
    {
        auto ae = a * e;
        auto af = a * f;
        auto be = b * e;
        auto bf = b * f;

        te[0] = c * e;
        te[4] = -c * f;
        te[8] = d;

        te[1] = af + be * d;
        te[5] = ae - bf * d;
        te[9] = -b * c;

        te[2] = bf - ae * d;
        te[6] = be + af * d;
        te[10] = a * c;
        break;
    }
    case EulerOrder::YXZ:
    {
        auto ce = c * e;
        auto cf = c * f;
        auto de = d * e;
        auto df = d * f;

        te[0] = ce + df * b;
        te[4] = de * b - cf;
        te[8] = a * d;

        te[1] = a * f;
        te[5] = a * e;
        te[9] = -b;

        te[2] = cf * b - de;
        te[6] = df + ce * b;
        te[10] = a * c;
        break;
    }
    case EulerOrder::ZXY:
    {
        auto ce = c * e;
        auto cf = c * f;
        auto de = d * e;
        auto df = d * f;

        te[0] = ce - df * b;
        te[4] = -a * f;
        te[8] = de + cf * b;

        te[1] = cf + de * b;
        te[5] = a * e;
        te[9] = df - ce * b;

        te[2] = -a * d;
        te[6] = b;
        te[10] = a * c;
        break;
    }
    case EulerOrder::ZYX:
    {
        auto ae = a * e;
        auto af = a * f;
        auto be = b * e;
        auto bf = b * f;

        te[0] = c * e;
        te[4] = be * d - af;
        te[8] = ae * d + bf;

        te[1] = c * f;
        te[5] = bf * d + ae;
        te[9] = af * d - be;

        te[2] = -d;
        te[6] = b * c;
        te[10] = a * c;
        break;
    }
    case EulerOrder::YZX:
    {
        auto ac = a * c;
        auto ad = a * d;
        auto bc = b * c;
        auto bd = b * d;

        te[0] = c * e;
        te[4] = bd - ac * f;
        te[8] = bc * f + ad;

        te[1] = f;
        te[5] = a * e;
        te[9] = -b * e;

        te[2] = -d * e;
        te[6] = ad * f + bc;
        te[10] = ac - bd * f;
        break;
    }
    case EulerOrder::XZY:
    {
        auto ac = a * c;
        auto ad = a * d;
        auto bc = b * c;
        auto bd = b * d;

        te[0] = c * e;
        te[4] = -f;
        te[8] = d * e;

        te[1] = ac * f + bd;
        te[5] = a * e;
        te[9] = ad * f - bc;

        te[2] = bc * f - ad;
        te[6] = b * e;
        te[10] = bd * f + ac;
        break;
    }
    default:
        assert(false);
    }

    // bottom row
    te[3] = 0;
    te[7] = 0;
    te[11] = 0;

    // last column
    te[12] = 0;
    te[13] = 0;
    te[14] = 0;
    te[15] = 1;

    ret.at<TFloat>(0, 0) = te[0];
    ret.at<TFloat>(0, 1) = te[1];
    ret.at<TFloat>(0, 2) = te[2];
    ret.at<TFloat>(1, 0) = te[4];
    ret.at<TFloat>(1, 1) = te[5];
    ret.at<TFloat>(1, 2) = te[6];
    ret.at<TFloat>(2, 0) = te[8];
    ret.at<TFloat>(2, 1) = te[9];
    ret.at<TFloat>(2, 2) = te[10];

    return ret;
}

template <typename TFloat>
TFloat radToDeg(TFloat rad)
{
    return rad / M_PI * 180;
}

template <typename TFloat>
TFloat degToRad(TFloat deg)
{
    return deg / 180 * M_PI;
}

namespace LibExif
{
    struct ExifLoaderDeleter
    {
        void operator()(ExifLoader* ld)
        {
            exif_loader_unref(ld);
        }
    };
    using ExifLoaderPtr = std::unique_ptr<ExifLoader, ExifLoaderDeleter>;
} // namespace LibExif

cv::detail::CameraParams createCamera(double focal, double aspect, double ppx, double ppy, cv::Mat R, cv::Mat t, bool useDouble = false)
{
    cv::detail::CameraParams ret;
    ret.focal = focal;
    ret.aspect = aspect;
    ret.ppx = ppx;
    ret.ppy = ppy;
    t.convertTo(ret.t, CV_32F);
    if (!useDouble)
    {
        R.convertTo(ret.R, CV_32F);
    }
    else
    {
        ret.R = R;
    }
    return ret;
}

struct CameraMergeState
{
    cv::detail::CameraParams sensor;
    cv::detail::CameraParams cv;

};

std::vector<std::string> splitMatrixStrItems(std::string_view sv)
{
    std::vector<std::string> ret;
    auto pos = sv.find(",");
    while (pos != sv.npos)
    {
        ret.emplace_back(sv.substr(0, pos));
        sv = sv.substr(pos + 1);
        pos = sv.find(",");
    }
    ret.emplace_back(sv);

    return ret;
}

cv::Mat parseMatrixStr(std::string_view sv)
{
    sv = sv.substr(1, sv.size() - 2);
    auto items = splitMatrixStrItems(sv);
    auto len = (int)std::sqrt(items.size());
    cv::Mat ret(cv::Size(len, len), CV_64F);
    for (auto y = 0; y < len; y++)
    {
        for (auto x = 0; x < len; x++)
        {
            ret.at<double>(x, y) = std::strtod(items.at(y * len + x).c_str(), nullptr);
        }
    }
    return ret;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        return -1;
    }

    std::transform(
        std::filesystem::directory_iterator(std::filesystem::path(argv[1])),
        std::filesystem::directory_iterator{}, std::back_inserter(img_names),
        [](const std::filesystem::directory_entry& de) {
            return de.path().string();
        });

    std::sort(std::begin(img_names), std::end(img_names), [](const std::string& a, const std::string& b) {
        auto pa = std::filesystem::path(a);
        auto pb = std::filesystem::path(b);

        auto namea = pa.filename().string();
        auto nameb = pb.filename().string();

        return std::strtol(namea.c_str(), nullptr, 10) < strtol(nameb.c_str(), nullptr, 10);
        });

    std::vector<cv::detail::CameraParams> camParamsFromSensor;

    std::transform(
        std::begin(img_names),
        std::end(img_names), std::back_inserter(camParamsFromSensor),
        [](const std::string& path) {
            LibExif::ExifLoaderPtr loader;
            loader.reset(exif_loader_new());
            exif_loader_write_file(loader.get(), path.c_str());
            auto data = exif_loader_get_data(loader.get());
            LOGLN("image: " << path);
            cv::detail::CameraParams ret{};
            auto getMatrix = [](ExifEntry* ee, void* user_data) {
                char buf[1024];
                if (ee->tag == EXIF_TAG_IMAGE_DESCRIPTION)
                {
                    exif_entry_get_value(ee, buf, std::end(buf) - std::begin(buf) - 1);
                    std::string_view sv(buf);
                    auto pos = sv.find(";");
                    auto projectionMatrixStr = sv.substr(0, pos);
                    pos++;
                    sv = sv.substr(pos);
                    pos = sv.find(";");
                    auto viewMatrixStr = sv.substr(0, pos);
                    pos++;
                    sv = sv.substr(pos);
                    auto kMatrixStr = sv;

                    auto projMatrix = parseMatrixStr(projectionMatrixStr);
                    auto viewMatrix = parseMatrixStr(viewMatrixStr);
                    cv::Mat camTransformMatrix = viewMatrix.inv();
                    auto kMatrix = parseMatrixStr(kMatrixStr);

                    std::cout << "projMatrix: " << projMatrix << std::endl <<
                        "viewMatrix: " << viewMatrix << std::endl <<
                        "cameraTransformMatrix" << camTransformMatrix << std::endl <<
                        "kMatrix: " << kMatrix << std::endl;

                    auto&& param = *reinterpret_cast<cv::detail::CameraParams*>(user_data);
                    param.aspect = 1.0;
                    param.focal = kMatrix.at<double>(0, 0);
                    param.ppx = kMatrix.at<double>(1, 2);
                    param.ppy = kMatrix.at<double>(0, 2);
                    param.R = cv::Mat(cv::Size(3, 3), CV_64F);
                    param.R.at<double>(0, 0) = camTransformMatrix.at<double>(0, 0);
                    param.R.at<double>(0, 1) = camTransformMatrix.at<double>(0, 1);
                    param.R.at<double>(0, 2) = camTransformMatrix.at<double>(0, 2);
                    param.R.at<double>(1, 0) = camTransformMatrix.at<double>(1, 0);
                    param.R.at<double>(1, 1) = camTransformMatrix.at<double>(1, 1);
                    param.R.at<double>(1, 2) = camTransformMatrix.at<double>(1, 2);
                    param.R.at<double>(2, 0) = camTransformMatrix.at<double>(2, 0);
                    param.R.at<double>(2, 1) = camTransformMatrix.at<double>(2, 1);
                    param.R.at<double>(2, 2) = camTransformMatrix.at<double>(2, 2);
                    param.t.at<double>(0, 0) = camTransformMatrix.at<double>(3, 0);
                    param.t.at<double>(1, 0) = camTransformMatrix.at<double>(3, 1);
                    param.t.at<double>(2, 0) = camTransformMatrix.at<double>(3, 2);
                    
                    std::cout << "t: " << param.t << std::endl <<
                        "R: " << param.R << std::endl;
                    //double alpha, beta, gamma;
                    //auto pos = sv.find("value0:");
                    //sv = sv.substr(pos + 7);
                    //alpha = std::strtod(sv.data(), nullptr);
                    //pos = sv.find("value1:");
                    //sv = sv.substr(pos + 7);
                    //beta = std::strtod(sv.data(), nullptr);
                    //pos = sv.find("value2:");
                    //sv = sv.substr(pos + 7);
                    //gamma = std::strtod(sv.data(), nullptr);

                    //auto r = reinterpret_cast<cv::Mat*>(user_data);
                    //
                    ////x *= -1.0;
                    ////y *= -1.0;
                    ////z *= -1.0;
                    //cv::Vec3d vec(beta, alpha, gamma);
                    //*r = cv::Mat(vec);

                    //LOGLN("image rotation "
                    //    << ": pitch: " << radToDeg(vec[0]) << ", "
                    //    << ", yaw: " << radToDeg(vec[1]) << ", roll: " << radToDeg(vec[2]));


                }
            };
            exif_content_foreach_entry(data->ifd[EXIF_IFD_0], getMatrix, &ret);

            return ret;
        });
    EulerOrder order = EulerOrder::YXZ;

    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    Ptr<Feature2D> finder;
    if (features_type == "orb")
    {
        finder = ORB::create(500, 1.2, 8, 1, 0, 2, cv::ORB::HARRIS_SCORE, 40, 20);
    }
    else if (features_type == "akaze")
    {
        finder = AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        finder = xfeatures2d::SURF::create();
    }
#endif
    else if (features_type == "sift")
    {
        finder = SIFT::create();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(samples::findFile(img_names[i]));
        cv::Mat tmp;
        rotate(full_img, tmp, ROTATE_90_CLOCKWISE);
        full_img = tmp;
        tmp.release();
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            LOGLN("Can't open image " << img_names[i]);
            return -1;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            cv::resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        if (0)
        {
            computeImageFeatures(finder, img, features[i]);
            features[i].img_idx = i;
            LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());
        }

        cv::resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

    full_img.release();
    img.release();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOG("Pairwise matching");
#if ENABLE_LOG
    t = getTickCount();
#endif
    vector<MatchesInfo> pairwise_matches;
    Ptr<FeaturesMatcher> matcher;
    if (matcher_type == "affine")
        matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    else if (range_width == -1)
        matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
    else
        matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
    std::vector<std::string> removedNames;
    vector<int> indices;
    if (0)
    {
        for (auto iter = std::begin(features); iter != std::end(features);)
        {
            if (iter->keypoints.size() != 500)
            {
                auto idx = iter->img_idx;
                iter = features.erase(iter);
                removedNames.emplace_back(img_names[idx]);
            }
            else
            {
                iter++;
            }
        }

        auto orig_img_names = img_names;
        img_names.erase(std::remove_if(std::begin(img_names), std::end(img_names), [&](const std::string& s) {
            return std::find(std::begin(removedNames), std::end(removedNames), s) != std::end(removedNames);
            }),
            img_names.end());

        (*matcher)(features, pairwise_matches);
        matcher->collectGarbage();
        LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

        // Check if we should save matches graph
        if (save_graph)
        {
            LOGLN("Saving matches graph...");
            ofstream f(save_graph_to.c_str());
            f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
        }

        // Leave only images we are sure are from the same panorama
        indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
        vector<Mat> img_subset;
        vector<String> img_names_subset;
        vector<Size> full_img_sizes_subset;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            img_names_subset.push_back(img_names[indices[i]]);
            img_subset.push_back(images[indices[i]]);
            full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
        }

        auto orig_images = images;
        auto orig_full_img_sizes = full_img_sizes;
        images = img_subset;
        img_names = img_names_subset;
        full_img_sizes = full_img_sizes_subset;
    }


    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
    vector<CameraParams> camParamsFromCV;
    CameraParams minFocal;
    if (0 && num_images >= 2)
    {
        Ptr<HomographyBasedEstimator> estimator;
        estimator = makePtr<HomographyBasedEstimator>();

        //{

        //    std::vector<double> focals;
        //    estimateFocal(features, pairwise_matches, focals);
        //    camParamsFromCV.assign(num_images, CameraParams());
        //    for (int i = 0; i < num_images; ++i)
        //    {
        //        camParamsFromCV[i].focal = focals[i];
        //        camParamsFromCV[i].ppx = 0.5 * features[i].img_size.width;
        //        camParamsFromCV[i].ppy = 0.5 * features[i].img_size.height;
        //    }

        //}
        //for (auto i = 0; i < indices.size(); i++)
        //{
        //    auto&& c = camParamsFromSensor[indices[i]];
        //    camParamsFromCV[i].R = eulerAnglesToRotationMatrix(cv::Vec3d(c.R.at<double>(0), c.R.at<double>(1), c.R.at<double>(2)));
        //}

        if (!(*estimator)(features, pairwise_matches, camParamsFromCV))
        {
            cout << "Homography estimation failed.\n";
            return -1;
        }

        for (size_t i = 0; i < camParamsFromCV.size(); ++i)
        {
            Mat R;
            camParamsFromCV[i].R.convertTo(R, CV_32F);
            camParamsFromCV[i].R = R;
        }

        Ptr<detail::BundleAdjusterBase> adjuster;
        if (ba_cost_func == "reproj")
            adjuster = makePtr<detail::BundleAdjusterReproj>();
        else if (ba_cost_func == "ray")
            adjuster = makePtr<detail::BundleAdjusterRay>();
        else if (ba_cost_func == "affine")
            adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
        else if (ba_cost_func == "no")
            adjuster = makePtr<NoBundleAdjuster>();
        else
        {
            cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
            return -1;
        }
        adjuster->setConfThresh(conf_thresh);
        Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
        if (ba_refine_mask[0] == 'x')
            refine_mask(0, 0) = 1;
        if (ba_refine_mask[1] == 'x')
            refine_mask(0, 1) = 1;
        if (ba_refine_mask[2] == 'x')
            refine_mask(0, 2) = 1;
        if (ba_refine_mask[3] == 'x')
            refine_mask(1, 1) = 1;
        if (ba_refine_mask[4] == 'x')
            refine_mask(1, 2) = 1;
        adjuster->setRefinementMask(refine_mask);
        if (!(*adjuster)(features, pairwise_matches, camParamsFromCV))
        {
            cout << "Camera parameters adjusting failed.\n";
            return -1;
         }
        for (auto&& cam : camParamsFromCV)
        {
            Mat R;
            cam.R.convertTo(R, CV_64F);
            cam.R = R;
            std::cout << "camera cv: " << cam.R << std::endl;
        }
        minFocal = *std::min_element(std::begin(camParamsFromCV), std::end(camParamsFromCV), [](auto&& a, auto&& b) {
            return a.focal < b.focal;
            });
    }

    cv::Mat baseR = eulerAnglesToRotationMatrix(cv::Vec3d(0, 0, 0), order);
    cv::Mat lastBR;
    std::vector<CameraMergeState> cameraMergeStates;

    std::transform(std::begin(camParamsFromSensor), std::end(camParamsFromSensor), std::back_inserter(cameraMergeStates), [&](auto&& c)
        {
            return CameraMergeState{ c };
            auto BR = eulerAnglesToRotationMatrix(cv::Vec3d(c.R.at<double>(0), c.R.at<double>(1), c.R.at<double>(2)), order);
            if (lastBR.empty())
            {
                lastBR = BR;
            }
            Quaternion<double> q;
            q.setFromRotationMatrix<double>(BR);
            Quaternion<double> q2;
            q2.setFromAxisAngle(cv::Vec3d{ -1.0, 0, 0 }, degToRad(90.0));
            auto R = q.multiply(q2).toRotationMatrix();

            std::cout << "BR: " << R << std::endl;

            double focal, ppx, ppy;
            if (0 && num_images >= 2)
            {
                focal = minFocal.focal;
                ppx = minFocal.ppx;
                ppy = minFocal.ppy;
            }
            else
            {
                focal = c.focal / 0.0015 * work_scale;
                ppx = features.front().img_size.width * 0.5;
                ppy = features.front().img_size.height * 0.5;
            }
            return CameraMergeState{ createCamera(
                focal,
                1.0,
                ppx,
                ppy,
                R,
                cv::Mat(cv::Vec3f{0, 0, 0}), true) };
        });
    if (0 && num_images >= 2) 
    {
        auto firstCVR = camParamsFromCV.front().R;
        auto cvFirstIdx = indices.front();
        auto cvSensorR = cameraMergeStates[cvFirstIdx].sensor.R;
        auto rDiff = firstCVR * cvSensorR.t();


        for (auto&& cvCam : camParamsFromCV)
        {
            cvCam.R = cvCam.R * rDiff;
        }
        {
            int i = 0;
            for (auto idx : indices)
            {
                cameraMergeStates[idx].cv = camParamsFromCV[i];
                i++;
            }
        }

        auto firstCvIter = std::find_if(std::begin(cameraMergeStates), std::end(cameraMergeStates), [](auto&& c)
            {
                return c.cv.focal != 1;
            });


        for (auto&& c : cameraMergeStates)
        {
            cv::Vec3d degVec;
            if (c.cv.focal == 1)
            {
                degVec = cv::Vec3d{ 0 , 0, 0 };
                c.cv = c.sensor;
            }
            else
            {
                cv::Mat val = c.sensor.R * c.cv.R.t();
                auto vec = rotationMatrixToEulerAngles<double>(val, EulerOrder::ZYX);
                auto degVec = cv::Vec3d{ radToDeg(vec[0]), radToDeg(vec[1]), radToDeg(vec[2]) };
            }
            std::cout << "error val: " << degVec << std::endl;
        }
    }
    

    //if (firstCvIter != std::begin(cameraMergeStates))
    //{
    //    for (auto i = std::make_reverse_iterator(firstCvIter); i != std::rend(cameraMergeStates); i++)
    //    {
    //        auto&& cur = *i;
    //        auto next = *(i - 1);

    //        auto curMat = cur.sensor.R;
    //        auto nextMat = next.sensor.R;

    //        auto nextCVMat = next.cv.R;

    //        auto SCt = next.sensor.R * next.cv.R.t();

    //        auto BAt = curMat * nextMat.t();
    //        auto calibrated = BAt * nextCVMat;

    //        cur.cv = cur.sensor;
    //        cur.cv.R = calibrated;
    //        cur.cv.focal = minFocal.focal;
    //        cur.cv.ppx = minFocal.ppx;
    //        cur.cv.ppy = minFocal.ppy;
    //    }
    //}

    //for (auto i = std::begin(cameraMergeStates) + 1; i != std::end(cameraMergeStates); i++)
    //{
    //    auto&& cur = *i;

    //    if (cur.cv.focal != 1)
    //    {
    //        continue;
    //    }
    //    auto prev = *(i - 1);

    //    auto curMat = cur.sensor.R;
    //    auto prevMat = prev.sensor.R;

    //    auto prevCVMat = prev.cv.R;

    //    auto BAt = curMat * prevMat.t();

    //    auto calibrated = prevCVMat * BAt;

    //    cur.cv = cur.sensor;
    //    cur.cv.R = calibrated;
    //    cur.cv.focal = minFocal.focal;
    //    cur.cv.ppx = minFocal.ppx;
    //    cur.cv.ppy = minFocal.ppy;
    //}

    std::vector<cv::detail::CameraParams> cameras;
    std::transform(std::begin(cameraMergeStates), std::end(cameraMergeStates), std::back_inserter(cameras), [](auto&& s)
        {
            return s.sensor;
        });

    for (auto&& cam : cameras)
    {
        cv::Mat tmp;
        cam.R.convertTo(tmp, CV_32F);
        cam.R = tmp;
    }
    for (auto& cam : cameras)
    {
        auto zyx = rotationMatrixToEulerAngles<float>(cam.R, EulerOrder::ZYX);
        auto zyxDeg = cv::Vec3d(radToDeg(zyx[0]), radToDeg(zyx[1]), radToDeg(zyx[2]));
        std::cout << "rotationMatrix: " << cam.R << std::endl;
        std::cout << "rotation: " << zyxDeg << std::endl;

        std::cout << "translation: " << cam.t << std::endl;
    }
    camParamsFromCV.clear();
    camParamsFromSensor.clear();

    num_images = cameras.size();
    //full_img_sizes = orig_full_img_sizes;
    //images = orig_images;
    //img_names = orig_img_names;
    //orig_full_img_sizes.clear();
    //orig_images.clear();
    //orig_img_names.clear();

    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    t = getTickCount();
#endif

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    // Prepare images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            warper_creator = makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            warper_creator = makePtr<cv::TransverseMercatorWarper>();
    }

    if (!warper_creator)
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return 1;
    }

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0, 0) *= swa;
        K(0, 2) *= swa;
        K(1, 1) *= swa;
        K(1, 2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Compensating exposure...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    if (dynamic_cast<GainCompensator*>(compensator.get()))
    {
        GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
    {
        ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<BlocksCompensator*>(compensator.get()))
    {
        BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(expos_comp_nr_feeds);
        bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
        bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }

    compensator->feed(corners, images_warped, masks_warped);

    LOGLN("Compensating exposure, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Finding seams...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder)
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return 1;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << img_idx + 1);

        // Read image and resize it if necessary
        full_img = imread(samples::findFile(img_names[img_idx]));
        cv::Mat tmp;
        rotate(full_img, tmp, ROTATE_90_CLOCKWISE);
        full_img = tmp;
        tmp.release();

        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            cv::resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        cv::dilate(masks_warped[img_idx], dilated_mask, Mat());
        cv::resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;

        if (!blender && !timelapse)
        {
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f / blend_width);
                LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }
        else if (!timelapser && timelapse)
        {
            timelapser = Timelapser::createDefault(timelapse_type);
            timelapser->initialize(corners, sizes);
        }

        // Blend the current image
        if (timelapse)
        {
            timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
            String fixedFileName;
            size_t pos_s = String(img_names[img_idx]).find_last_of("/\\");
            if (pos_s == String::npos)
            {
                fixedFileName = "fixed_" + img_names[img_idx];
            }
            else
            {
                fixedFileName = "fixed_" + String(img_names[img_idx]).substr(pos_s + 1, String(img_names[img_idx]).length() - pos_s);
            }
            imwrite(fixedFileName, timelapser->getDst());
        }
        else
        {
            blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        }
    }

    if (!timelapse)
    {
        Mat result, result_mask;
        blender->blend(result, result_mask);

        LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

        imwrite(result_name, result);
    }

    return 0;
    }