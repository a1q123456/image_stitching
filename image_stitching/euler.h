#pragma once


template<typename TFloat>
cv::Vec<TFloat, 3> rotationMatrixToEulerAngles(cv::Mat R, EulerOrder order)
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
cv::Mat eulerAnglesToRotationMatrix(const cv::Vec<TFloat, 3>& euler, EulerOrder order)
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

    ret.template at<TFloat>(0, 0) = te[0];
    ret.template at<TFloat>(1, 0) = te[1];
    ret.template at<TFloat>(2, 0) = te[2];
    ret.template at<TFloat>(0, 1) = te[4];
    ret.template at<TFloat>(1, 1) = te[5];
    ret.template at<TFloat>(2, 1) = te[6];
    ret.template at<TFloat>(0, 2) = te[8];
    ret.template at<TFloat>(1, 2) = te[9];
    ret.template at<TFloat>(2, 2) = te[10];

    return ret;
}


