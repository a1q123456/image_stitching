#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include "serializer.h"

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
            ret.at<double>(y, x) = std::strtod(items.at(y * len + x).c_str(), nullptr);
        }
    }
    return ret;
}

std::string serializeMatrix(const cv::Mat& m)
{
    std::stringstream ss;

    ss << "[";
    for (auto r = 0; r < m.rows; r++)
    {
        for (auto c = 0; c < m.cols; c++)
        {
            if (m.type() == CV_32F)
            {
                ss << m.at<float>(r, c);
            }
            else if (m.type() == CV_64F)
            {
                ss << m.at<double>(r, c);
            }
            if (c == m.cols - 1)
            {
                ss << ";";
            }
            else
            {
                ss << ",";
            }
        }
    }
    ss << "]";
    return ss.str();
}

cv::Mat deserializeMatrix(std::string s)
{
    s = s.substr(1);
    std::vector<double> values;
    int nCols = 0, nRows = 0;

    const char* data = s.c_str();
    while (true)
    {
        double val;
        char* end;
        values.push_back(std::strtold(data, &end));
        data = end + 1;

        if (*end == ';')
        {
            if (nRows == 0)
            {
                nCols++;
            }
            nRows++;
        }
        else if (nRows == 0)
        {
            nCols++;
        }

        if (*data == ']')
        {
            break;
        }
    }

    cv::Mat ret = cv::Mat::eye(cv::Size(nCols, nRows), CV_32F);
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            ret.at<float>(i, j) = values[nCols * i + j];
        }
    }
    return ret;
}

void serializeCameraParams(const std::vector<cv::detail::CameraParams>& cams)
{
    std::fstream fs;
    fs.open("./cams.data", std::ios::out);
    for (auto&& c : cams)
    {
        fs << c.aspect << "@"
            << c.focal << "@"
            << c.ppx << "@"
            << c.ppy << "@"
            << serializeMatrix(c.t) << "@"
            << serializeMatrix(c.R) << std::endl;
    }
}

std::vector<cv::detail::CameraParams> deserializeCameraParams()
{
    std::vector<cv::detail::CameraParams> ret;
    std::fstream fs;
    fs.open("./cams.data", std::ios::in);
    std::string line;
    while (std::getline(fs, line))
    {
        auto pos = line.find("@");
        auto aspectStr = line.substr(0, pos);
        pos++;
        line = line.substr(pos);
        pos = line.find("@");
        auto focalStr = line.substr(0, pos);
        pos++;
        line = line.substr(pos);
        pos = line.find("@");
        auto ppxStr = line.substr(0, pos);
        pos++;
        line = line.substr(pos);
        pos = line.find("@");
        auto ppyStr = line.substr(0, pos);
        pos++;
        line = line.substr(pos);
        pos = line.find("@");
        auto tStr = line.substr(0, pos);
        pos++;
        auto RStr = line.substr(pos);

        cv::detail::CameraParams c;
        c.aspect = std::strtod(aspectStr.c_str(), nullptr);
        c.focal = std::strtod(focalStr.c_str(), nullptr);
        c.ppx = std::strtod(ppxStr.c_str(), nullptr);
        c.ppy = std::strtod(ppyStr.c_str(), nullptr);
        c.R = deserializeMatrix(RStr);
        c.t = deserializeMatrix(tStr);
        ret.emplace_back(c);
    }
    return ret;
}

void serializeIndices(const std::vector<int>& indicies)
{
    std::fstream fs;
    fs.open("./indices.data", std::ios::out);
    for (auto i : indicies)
    {
        fs << i << std::endl;
    }
}

std::vector<int> deserializeIndices()
{
    std::fstream fs;
    fs.open("./indices.data", std::ios::in);
    std::vector<int> ret;
    std::string line;
    while (std::getline(fs, line))
    {
        if (!line.empty())
        {
            ret.emplace_back(std::strtol(line.c_str(), nullptr, 10));
        }
    }
    return ret;
}
