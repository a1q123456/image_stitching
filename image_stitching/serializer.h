#pragma once

#include <sstream>

std::vector<std::string> splitMatrixStrItems(std::string_view sv);

cv::Mat parseMatrixStr(std::string_view sv);


std::string serializeMatrix(const cv::Mat& m);

cv::Mat deserializeMatrix(std::string s);

void serializeCameraParams(const std::vector<cv::detail::CameraParams>& cams);

std::vector<cv::detail::CameraParams> deserializeCameraParams();

void serializeIndices(const std::vector<int>& indicies);

std::vector<int> deserializeIndices();

