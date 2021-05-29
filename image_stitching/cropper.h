#pragma once

bool checkInteriorExterior(const cv::Mat& mask, const cv::Rect& croppingMask,
    int& top, int& bottom, int& left, int& right);

bool compareX(cv::Point a, cv::Point b);

bool compareY(cv::Point a, cv::Point b);

void crop(cv::Mat& source);
