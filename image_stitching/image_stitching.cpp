
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
#include <opencv2/stitching.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include <filesystem>
#include <libexif/exif-data.h>
#include <libexif/exif-loader.h>

#include <numeric>
#include <optional>

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#include "euler_order.h"
#include "quaternion.h"
#include "euler.h"

#include "cropper.h"
#include "serializer.h"

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = true;
double work_megapix = -1;
double seam_megapix = 0.1;
double compose_megapix = 0.4;
float conf_thresh = 0.95f;
#ifdef HAVE_OPENCV_XFEATURES2D
string features_type = "surf";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.32f;
#endif
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "reproj";
string ba_refine_mask = "_____";
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
int timelapse_type = Timelapser::CROP;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;
bool find_features = true;
bool serialize_data = true;

struct CaptureModeDesc {
    double x = 0;
    int totalImg = 0;
    double error = 0;
    double zError = 0;
    double angles[30];
    double startY = 0;
};

constexpr CaptureModeDesc captureModeDesc[]{
    { 0, 11, 1.0, 0.0, { 0, 0, } },
    { 34.0, 9, 1.2, 0.0, {0, 36, 75.8, 115.8, 155.8, 195.8, 235.8, 275.8, 315.8}, 4.1 },
    { 72, 4, 0, 0, { 0, 83, 180, 277 } },
    { -36, 9, 0.6, 0.0, {0, 36.7, 78, 117, 161.5, 200, 243, 279, 320}, 4.1 },
    { -72, 4, 0.0, 0.0, { 0, 83, 180, 277 } },
};

int getGroup(int idx)
{
    auto iter = std::begin(captureModeDesc);
    int curGroup = 0;
    while ((idx - iter->totalImg) >= 0)
    {
        curGroup++;
        idx -= iter->totalImg;
        iter++;
    }
    return curGroup;
}

int getGroupIdx(int idx, int group)
{
    auto groupIdx = idx - std::accumulate(std::begin(captureModeDesc), std::begin(captureModeDesc) + group, 0, [](int a, CaptureModeDesc d)
        {
            return a + d.totalImg;
        });
    return groupIdx;
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

auto getFov(const CameraParams& cam)
{
    auto fx = cam.K().at<double>(0, 0);
    auto ppx = cam.ppx;
    auto fy = cam.K().at<double>(1, 1);
    auto ppy = cam.ppy;

    auto fovX = 2 * std::atan(ppx / fx);
    auto fovY = 2 * std::atan(ppy / fy);

    return std::make_pair(fovX, fovY);
};

auto getGroupStartEnd(int group)
{
    auto from = std::accumulate(std::begin(captureModeDesc), std::begin(captureModeDesc) + group, 0, [](int a, const CaptureModeDesc& d)
        {
            return a + d.totalImg;
        });
    auto to = from + captureModeDesc[group].totalImg - 1;
    return std::make_pair(from, to);
}

auto getFieldRect(
    const std::vector<CameraParams>& cameras, 
    int groupId, 
    int index)
{
    cv::Rect srcRect{ };
    auto [srcFrom, srcTo] = getGroupStartEnd(groupId);
    auto desc = captureModeDesc[groupId];
    srcRect.y = desc.x;
    srcRect.x = (M_PI * 2) / desc.totalImg * (index - srcFrom);
    auto [w, h] = getFov(cameras[index]);
    srcRect.width = w;
    srcRect.height = h;

    return srcRect;
}

std::vector<int> myLeaveBiggestComponent(
    const std::vector<CameraParams>& cameras,
    std::vector<ImageFeatures>& features, 
    std::vector<MatchesInfo>& pairwise_matches,
    float conf_threshold)
{
    const int num_images = static_cast<int>(features.size());


    DisjointSets comps(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            auto pm = pairwise_matches[i * num_images + j];
            if (pm.confidence < conf_threshold)
                continue;


            int comp1 = comps.findSetByElem(i);
            int comp2 = comps.findSetByElem(j);
            if (comp1 != comp2)
                comps.mergeSets(comp1, comp2);
        }
    }

    int max_comp = static_cast<int>(std::max_element(comps.size.begin(), comps.size.end()) - comps.size.begin());

    std::vector<int> indices;
    std::vector<int> indices_removed;
    for (int i = 0; i < num_images; ++i)
        if (comps.findSetByElem(i) == max_comp)
            indices.push_back(i);
        else
            indices_removed.push_back(i);

    std::vector<ImageFeatures> features_subset;
    std::vector<MatchesInfo> pairwise_matches_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        features_subset.push_back(features[indices[i]]);
        for (size_t j = 0; j < indices.size(); ++j)
        {
            pairwise_matches_subset.push_back(pairwise_matches[indices[i] * num_images + indices[j]]);
            pairwise_matches_subset.back().src_img_idx = static_cast<int>(i);
            pairwise_matches_subset.back().dst_img_idx = static_cast<int>(j);
        }
    }

    if (static_cast<int>(features_subset.size()) == num_images)
        return indices;

    LOG("Removed some images, because can't match them or there are too similar images: (");
    LOG(indices_removed[0] + 1);
    for (size_t i = 1; i < indices_removed.size(); ++i)
        LOG(", " << indices_removed[i] + 1);
    LOGLN(").");
    LOGLN("Try to decrease the match confidence threshold and/or check if you're stitching duplicates.");

    features = features_subset;
    pairwise_matches = pairwise_matches_subset;

    return indices;
}


int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        return -1;
    }

    //{
    //    auto s = Stitcher::create(Stitcher::PANORAMA);
    //    std::vector<cv::Mat> in;
    //    in.emplace_back(imread(samples::findFile("./t.jpg")));
    //    in.emplace_back(imread(samples::findFile("./ct.jpg")));
    //    in.emplace_back(imread(samples::findFile("./c.jpg")));
    //    in.emplace_back(imread(samples::findFile("./cd.jpg")));
    //    in.emplace_back(imread(samples::findFile("./d.jpg")));

    //    cv::Mat ret;
    //    auto status = s->stitch(in, ret);
    //    if (status != Stitcher::OK)
    //    {
    //        return -1;
    //    }
    //    imwrite("my_result2.jpg", ret);
    //}

    std::vector<std::filesystem::directory_entry> img_paths;
    std::copy_if(std::filesystem::directory_iterator(std::filesystem::path(argv[1])),
        std::filesystem::directory_iterator{},
        std::back_inserter(img_paths), [](const std::filesystem::directory_entry& de) {
            return de.is_regular_file() &&
                (de.path().extension() == ".jpg" ||
                    de.path().extension() == ".jpeg" ||
                    de.path().extension() == ".JPG" ||
                    de.path().extension() == ".JPEG" ||
                    de.path().extension() == ".png" ||
                    de.path().extension() == ".PNG");
        });

    std::transform(
        std::begin(img_paths),
        std::end(img_paths),
        std::back_inserter(img_names),
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
    bool isPortrait = false;

    std::transform(
        std::begin(img_names),
        std::end(img_names), std::back_inserter(camParamsFromSensor),
        [&](const std::string& path) {
            LibExif::ExifLoaderPtr loader;
            loader.reset(exif_loader_new());
            exif_loader_write_file(loader.get(), path.c_str());
            auto data = exif_loader_get_data(loader.get());
            LOGLN("image: " << path);
            struct CamParamState
            {
                cv::detail::CameraParams ret{};
                bool picIsPortrait = false;
                std::function<cv::Vec3d()> getRotation;
                std::string path;
            };

            struct CalcRotation
            {
                int idx = 0;

                CalcRotation(const CamParamState& s, const std::vector<std::string>& img_names)
                {
                    auto iter = std::find(std::begin(img_names), std::end(img_names), s.path);
                    assert(iter != std::end(img_names));
                    idx = std::distance(std::begin(img_names), iter);
                }

                cv::Vec3d operator()()
                {
                    constexpr auto groups = std::end(captureModeDesc) - std::begin(captureModeDesc);
                    auto iter = std::begin(captureModeDesc);
                    int curGroup = 0;
                    int tmpIdx = idx;
                    while ((tmpIdx - iter->totalImg) >= 0)
                    {
                        curGroup++;
                        tmpIdx -= iter->totalImg;
                        iter++;
                    }
                    auto desc = captureModeDesc[curGroup];
                    auto groupIdx = idx - std::accumulate(std::begin(captureModeDesc), std::begin(captureModeDesc) + curGroup, 0, [](int a, CaptureModeDesc d)
                        {
                            return a + d.totalImg;
                        });
                    auto isSetEveryAngle = desc.angles[1] != 0;
                    double angleValue;
                    if (isSetEveryAngle)
                    {
                        angleValue = groupIdx * desc.error + desc.angles[groupIdx];
                    }
                    else
                    {
                        angleValue = groupIdx* (360.0 / desc.totalImg + desc.error);
                    }
                    angleValue += desc.startY;
                    if (angleValue > 180)
                    {
                        angleValue -= 360;
                    }
                    return cv::Vec3d{ 
                        degToRad(desc.x), 
                        degToRad(angleValue),
                        degToRad(desc.zError) };
                }
            };

            CamParamState state;
            state.path = path;
            state.getRotation = CalcRotation{ state, img_names };

            auto getMatrix = [](ExifEntry* ee, void* user_data) {
                char buf[1024];
                if (ee->tag == EXIF_TAG_IMAGE_DESCRIPTION)
                {
                    auto&& state = *reinterpret_cast<CamParamState*>(user_data);
                    exif_entry_get_value(ee, buf, std::end(buf) - std::begin(buf) - 1);
                    std::string_view sv(buf);

                    auto pos = sv.find(";");
                    auto isPortraitStr = std::string(sv.substr(0, pos));
                    pos++;
                    sv = sv.substr(pos);
                    pos = sv.find(";");
                    auto compassAngleStr = std::string(sv.substr(0, pos));
                    pos++;
                    sv = sv.substr(pos);
                    pos = sv.find(";");
                    auto projectionMatrixStr = sv.substr(0, pos);
                    pos++;
                    sv = sv.substr(pos);
                    pos = sv.find(";");
                    auto viewMatrixStr = sv.substr(0, pos);
                    pos++;
                    sv = sv.substr(pos);
                    pos = sv.find(";");
                    auto cameraTransformMatrixStr = sv.substr(0, pos);
                    pos++;
                    sv = sv.substr(pos);
                    auto kMatrixStr = sv;

                    auto projMatrix = parseMatrixStr(projectionMatrixStr);
                    auto viewMatrix = parseMatrixStr(viewMatrixStr);
                    auto camTransformMatrix = parseMatrixStr(cameraTransformMatrixStr);
                    auto kMatrix = parseMatrixStr(kMatrixStr);
                    state.picIsPortrait = (bool)std::strtol(isPortraitStr.c_str(), nullptr, 10);

                    //std::cout << "projMatrix: " << projMatrix << std::endl <<
                    //    "viewMatrix: " << viewMatrix << std::endl <<
                    //    "cameraTransformMatrix" << camTransformMatrix << std::endl <<
                    //    "kMatrix: " << kMatrix << std::endl;

                    auto&& param = state.ret;
                    param.aspect = 1.0;
                    param.focal = kMatrix.at<double>(1, 1);
                    if (state.picIsPortrait)
                    {
                        param.ppx = kMatrix.at<double>(1, 2);
                        param.ppy = kMatrix.at<double>(0, 2);
                    }
                    else
                    {
                        param.ppx = kMatrix.at<double>(0, 2);
                        param.ppy = kMatrix.at<double>(1, 2);
                    }
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
                    param.t = cv::Mat(cv::Vec3d());
                    param.t.at<double>(0, 0) = camTransformMatrix.at<double>(0, 3);
                    param.t.at<double>(1, 0) = camTransformMatrix.at<double>(1, 3);
                    param.t.at<double>(2, 0) = camTransformMatrix.at<double>(2, 3);
                    cv::Mat scaler = cv::Mat::eye(cv::Size(3, 3), CV_64F);
                    scaler.at<double>(0, 0) = 1.0;
                    scaler.at<double>(1, 1) = 1.0;
                    scaler.at<double>(2, 2) = 1.0;
                    cv::Mat R = param.R;
                    param.R = scaler * R;
                    Quaternion<double> q;
                    Quaternion<double> q2;
                    //std::cout << "OrigR: " << param.R << std::endl;
                    q.setFromRotationMatrix<double>(param.R);
                    //std::cout << "RQ: " << q.toRotationMatrix() << std::endl;
                    if (state.picIsPortrait)
                    {
                        q2.set(q.y(), q.x(), -q.z(), q.w());
                    }
                    else
                    {
                        q2.set(-q.x(), q.y(), -q.z(), q.w());
                    }
                    //auto euler = rotationMatrixToEulerAngles<double>(q2.toRotationMatrix(), EulerOrder::YXZ);
                    //euler = cv::Vec3d{ radToDeg(euler[0]), radToDeg(euler[1]), radToDeg(euler[2]) };
                    //
                    //auto euler = state.getRotation();
                    //q2.setFromEuler(euler, EulerOrder::YXZ);
                    //std::cout << "euler: " << euler << std::endl;
                    //q2.setFromEuler(euler, EulerOrder::ZYX);
                    //{
                    //    auto R = q2.toRotationMatrix();
                    //    auto order = EulerOrder::YXZ;
                    //    auto euler = rotationMatrixToEulerAngles<double>(R, order);
                    //    euler[1] = std::strtod(compassAngleStr.c_str(), nullptr);
                    //    R = eulerAnglesToRotationMatrix(euler, order);
                    //    std::cout << "R: " << R << std::endl;
                    //    q.setFromRotationMatrix<double>(R);
                    //}
                    q = q2;

                    //std::cout << "Q: " << q << std::endl;
                    param.R = q.toRotationMatrix();

                    //std::cout << "cameraR: " << param.R << std::endl
                    //    << "cameraT: " << param.t << std::endl
                    //    << "K: " << param.K() << std::endl
                    //    << "scaler: " << scaler << std::endl;
                }
            };
            exif_content_foreach_entry(data->ifd[EXIF_IFD_0], getMatrix, &state);
            isPortrait = state.picIsPortrait;
            return state.ret;
        });

    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    Ptr<Feature2D> finder;
    if (features_type == "orb")
    {
        finder = ORB::create(4000, 1.2, 8, 1, 0, 2, cv::ORB::HARRIS_SCORE, 40, 20);
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
        finder = SIFT::create(/*0, 3, 0.04, 2, 1.6*/);
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }

    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(samples::findFile(img_names[i]));
        cv::Mat tmp;
        if (isPortrait)
        {
            rotate(full_img, tmp, ROTATE_90_CLOCKWISE);
            full_img = tmp;
        }
        else
        {
            rotate(full_img, tmp, ROTATE_180);
            full_img = tmp;
        }
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

        if (find_features && serialize_data)
        {
            computeImageFeatures(finder, img, features[i]);
            features[i].img_idx = i;
        }

        LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

        cv::resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);

        images[i] = img.clone();
    }

    full_img.release();
    img.release();
    for (auto&& cam : camParamsFromSensor)
    {
        cv::Mat tmp;
        cam.R.convertTo(tmp, CV_32F);
        cam.R = tmp;

        cam.t.convertTo(tmp, CV_32F);
        cam.t = tmp;

        cam.focal *= work_scale;
        cam.ppx *= work_scale;
        cam.ppy *= work_scale;
    }
    std::vector<cv::detail::CameraParams> cameras = camParamsFromSensor;
    if (find_features)
    {
        vector<MatchesInfo> pairwise_matches;
        Ptr<FeaturesMatcher> matcher;
        if (matcher_type == "affine")
            matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
        else if (range_width == -1)
            matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
        else
            matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
        vector<int> indices;
        if (serialize_data)
        {
            (*matcher)(features, pairwise_matches);
        }
        matcher->collectGarbage();

        std::vector<cv::detail::CameraParams> camera_params_subset;

        if (serialize_data)
        {
            indices = myLeaveBiggestComponent(camParamsFromSensor, features, pairwise_matches, conf_thresh);
            // indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
        }
        if (indices.size() >= 2 || !serialize_data)
        {
            vector<Mat> img_subset;
            vector<String> img_names_subset;
            vector<Size> full_img_sizes_subset;
            if (!serialize_data)
            {
                indices = deserializeIndices();
            }
            for (size_t i = 0; i < indices.size(); ++i)
            {
                img_names_subset.push_back(img_names[indices[i]]);
                img_subset.push_back(images[indices[i]]);
                full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
                camera_params_subset.push_back(cameras[indices[i]]);
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
            if (serialize_data)
            {
                if (!(*adjuster)(features, pairwise_matches, camera_params_subset))
                {
                    cout << "Camera parameters adjusting failed.\n";
                    return -1;
                }
                serializeCameraParams(camera_params_subset);
                serializeIndices(indices);
            }
            else
            {
                camera_params_subset = deserializeCameraParams();
            }
            if (do_wave_correct)
            {
                vector<Mat> rmats;
                for (size_t i = 0; i < camera_params_subset.size(); ++i)
                    rmats.push_back(camera_params_subset[i].R.clone());
                waveCorrect(rmats, wave_correct);
                for (size_t i = 0; i < camera_params_subset.size(); ++i)
                    camera_params_subset[i].R = rmats[i];
            }
            {
                auto origEuler = rotationMatrixToEulerAngles<float>(cameras[indices.front()].R, EulerOrder::YXZ);
                auto firstEuler = rotationMatrixToEulerAngles<float>(camera_params_subset.front().R, EulerOrder::YXZ);
                float deltaY = origEuler[1] - firstEuler[1];

                for (int i = 0; i < camera_params_subset.size(); i++)
                {
                    auto&& cam = camera_params_subset[i];
                    auto euler = rotationMatrixToEulerAngles<float>(cam.R, EulerOrder::YXZ);
                    auto dEuler = cv::Vec3f{ radToDeg(euler[0]), radToDeg(euler[1]), radToDeg(euler[2]) };
                    euler[1] += deltaY;
                    std::cout << "idx: " << indices[i] << ", ba euler: " << euler << std::endl;
                    //cam.R = eulerAnglesToRotationMatrix(euler, EulerOrder::YXZ);
                }


                cameras = camera_params_subset;
                img_names = img_names_subset;
                num_images = img_names.size();

            }
#if 1


            if (0)
            {

                std::vector<std::optional<CameraParams>> refined_cams;
                refined_cams.resize(cameras.size());
                for (auto i = 0; i < indices.size(); i++)
                {
                    refined_cams[indices[i]] = camera_params_subset[i];
                }

                auto find_nearest_index = [](const std::vector<std::optional<CameraParams>>& cams, int cur)
                {
                    auto iter = std::begin(captureModeDesc);
                    auto tmpIdx = cur;
                    int groupIdx = 0;
                    while ((tmpIdx - iter->totalImg) >= 0)
                    {
                        groupIdx++;
                        tmpIdx -= iter->totalImg;
                        iter++;
                    }
                    auto from = std::accumulate(std::begin(captureModeDesc), std::begin(captureModeDesc) + groupIdx, 0, [](int a, const CaptureModeDesc& d)
                        {
                            return a + d.totalImg;
                        });
                    auto to = from + iter->totalImg - 1;
                    auto desc = captureModeDesc[groupIdx];
                    int i = cur, j = cur;
                    while (!cams[i] && !cams[j])
                    {
                        if (i != to)
                        {
                            i++;
                        }
                        if (j != from)
                        {
                            j--;
                        }
                        if (i == to && j == from)
                        {
                            break;
                        }
                    }
                    if (cams[i])
                    {
                        return i;
                    }
                    if (cams[j])
                    {
                        return j;
                    }
                    // fallback
                    i = cur;
                    j = cur;
                    while (!cams[i] && !cams[j])
                    {
                        if (i != cams.size() - 1)
                        {
                            i++;
                        }
                        if (j != 0)
                        {
                            j--;
                        }
                    }
                    return cams[i] ? i : j;
                };

                auto cam_result = cameras;
                for (int i = 0; i < refined_cams.size(); i++)
                {
                    auto result = refined_cams[i];
                    if (!result)
                    {
                        auto cur = cameras[i];
                        auto nearest_idx = find_nearest_index(refined_cams, i);
                        auto cur_R = cur.R;
                        auto ref_R = cameras[nearest_idx].R;
                        auto base_R = refined_cams[nearest_idx].value().R;

                        cv::Vec3f cur_E, ref_E,base_E;
                        cur_E = rotationMatrixToEulerAngles<float>(cur_R, EulerOrder::YXZ);
                        ref_E = rotationMatrixToEulerAngles<float>(ref_R, EulerOrder::YXZ);
                        base_E = rotationMatrixToEulerAngles<float>(base_R, EulerOrder::YXZ);
                        auto deltaY = cur_E[1] - ref_E[1];
                        auto deltaX = cur_E[0] - ref_E[0];
                        base_E[1] += deltaY;
                        base_E[0] += deltaX;
                        base_E[2] = 0;
                        cur.R = eulerAnglesToRotationMatrix(base_E, EulerOrder::YXZ);

                        //Quaternion<float> base_Q, cur_Q, ref_Q;
                        //cur_Q.setFromRotationMatrix<float>(cur_R);
                        //ref_Q.setFromRotationMatrix<float>(ref_R);
                        //cur_Q.multiply(ref_Q.conjugate());
                        //base_Q.setFromRotationMatrix<float>(base_R);
                        //base_Q.multiply(cur_Q);

                        //cur.R = base_Q.toRotationMatrix();
                        result = cur;
                    }
                    cam_result[i] = result.value();
                }
                cameras = cam_result;
                //for (int i = 0; i < cameras.size(); i++)
                //{
                //    cameras[i] = refined_cams[i].value();
                //}
                //cameras[18] = camParamsFromSensor[18];
                //cameras[19] = camParamsFromSensor[19];
                //cameras[20] = camParamsFromSensor[20];
                //cameras[21] = camParamsFromSensor[21];
            }
            
#endif
        }
    }




    //camParamsFromSensor.clear();

    //num_images = cameras.size();
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



    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    auto t = getTickCount();
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
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << img_idx + 1);

        // Read image and resize it if necessary
        full_img = imread(samples::findFile(img_names[img_idx]));
        cv::Mat tmp;
        if (isPortrait)
        {
            rotate(full_img, tmp, ROTATE_90_CLOCKWISE);
            full_img = tmp;
        }
        else
        {
            rotate(full_img, tmp, ROTATE_180);
            full_img = tmp;
        }
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