#pragma once

#include <opencv2/core/utility.hpp>

template<typename TFloat>
class Quaternion 
{
	template<typename TFloat>
	friend std::ostream& operator <<(std::ostream& s, const Quaternion<TFloat>& q);
private:
	TFloat _x, _y, _z, _w;
public:
	Quaternion(TFloat x = 0, TFloat y = 0, TFloat z = 0, TFloat w = 1) {
		_x = x;
		_y = y;
		_z = z;
		_w = w;

	}

	TFloat x() const { return _x; }
	TFloat y() const { return _y; }
	TFloat z() const { return _z; }
	TFloat w() const { return _w; }

	static Quaternion slerp(const Quaternion& qa, const Quaternion& qb, Quaternion& qm, TFloat t) {

		return qm.copy(qa).slerp(qb, t);

	}

	static void slerpFlat(
		TFloat* dst, 
		int dstOffset, 
		const TFloat* src0,
		int srcOffset0, 
		const TFloat* src1,
		int srcOffset1, 
		TFloat t) {

		// fuzz-free, array-based Quaternion SLERP operation

		TFloat x0 = src0[srcOffset0 + 0],
			y0 = src0[srcOffset0 + 1],
			z0 = src0[srcOffset0 + 2],
			w0 = src0[srcOffset0 + 3];

		TFloat x1 = src1[srcOffset1 + 0],
			y1 = src1[srcOffset1 + 1],
			z1 = src1[srcOffset1 + 2],
			w1 = src1[srcOffset1 + 3];

		if (t == 0) {

			dst[dstOffset + 0] = x0;
			dst[dstOffset + 1] = y0;
			dst[dstOffset + 2] = z0;
			dst[dstOffset + 3] = w0;
			return;

		}

		if (t == 1) {

			dst[dstOffset + 0] = x1;
			dst[dstOffset + 1] = y1;
			dst[dstOffset + 2] = z1;
			dst[dstOffset + 3] = w1;
			return;

		}

		if (w0 != w1 || x0 != x1 || y0 != y1 || z0 != z1) {

			auto s = TFloat(1) - t;
			TFloat cos = x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1,
				dir = (cos >= 0 ? 1 : -1),
				sqrSin = 1 - cos * cos;

			// Skip the Slerp for tiny steps to avoid numeric problems:
			if (sqrSin > std::numeric_limits<TFloat>::epsilon()) {

				auto sin = std::sqrt(sqrSin);
				auto len = std::atan2(sin, cos * dir);

				s = std::sin(s * len) / sin;
				t = std::sin(t * len) / sin;

			}

			auto tDir = t * dir;

			x0 = x0 * s + x1 * tDir;
			y0 = y0 * s + y1 * tDir;
			z0 = z0 * s + z1 * tDir;
			w0 = w0 * s + w1 * tDir;

			// Normalize in case we just did a lerp:
			if (s == TFloat(1) - t) {

				auto f = 1 / std::sqrt(x0 * x0 + y0 * y0 + z0 * z0 + w0 * w0);

				x0 *= f;
				y0 *= f;
				z0 *= f;
				w0 *= f;

			}

		}

		dst[dstOffset] = x0;
		dst[dstOffset + 1] = y0;
		dst[dstOffset + 2] = z0;
		dst[dstOffset + 3] = w0;

	}

	static TFloat* multiplyQuaternionsFlat(
		TFloat* dst, 
		int dstOffset, 
		const TFloat* src0, 
		int srcOffset0, 
		const TFloat* src1, 
		int srcOffset1) 
	{

		auto x0 = src0[srcOffset0];
		auto y0 = src0[srcOffset0 + 1];
		auto z0 = src0[srcOffset0 + 2];
		auto w0 = src0[srcOffset0 + 3];

		auto x1 = src1[srcOffset1];
		auto y1 = src1[srcOffset1 + 1];
		auto z1 = src1[srcOffset1 + 2];
		auto w1 = src1[srcOffset1 + 3];

		dst[dstOffset] = x0 * w1 + w0 * x1 + y0 * z1 - z0 * y1;
		dst[dstOffset + 1] = y0 * w1 + w0 * y1 + z0 * x1 - x0 * z1;
		dst[dstOffset + 2] = z0 * w1 + w0 * z1 + x0 * y1 - y0 * x1;
		dst[dstOffset + 3] = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1;

		return dst;

	}

	Quaternion& set(TFloat x, TFloat y, TFloat z, TFloat w) {

		_x = x;
		_y = y;
		_z = z;
		_w = w;

		return *this;

	}

	void clone() {

	//	return new this.constructor(this._x, this._y, this._z, this._w);

	}

	Quaternion& copy(const Quaternion& quaternion) {
		_x = quaternion._x;
		_y = quaternion._y;
		_z = quaternion._z;
		_w = quaternion._w;
		return *this;
	}

	template<typename TVecFloat>
	Quaternion& setFromEuler(cv::Vec<TVecFloat, 3> euler, EulerOrder order) {

		TVecFloat x = euler[0], y = euler[1], z = euler[2];

		// http://www.mathworks.com/matlabcentral/fileexchange/
		// 	20696-function-to-convert-between-dcm-euler-angles-quaternions-and-euler-vectors/
		//	content/SpinCalc.m

		using std::cos;
		using std::sin;

		auto c1 = cos(x / 2);
		auto c2 = cos(y / 2);
		auto c3 = cos(z / 2);

		auto s1 = sin(x / 2);
		auto s2 = sin(y / 2);
		auto s3 = sin(z / 2);

		switch (order) {

		case EulerOrder::XYZ:
			this->_x = s1 * c2 * c3 + c1 * s2 * s3;
			this->_y = c1 * s2 * c3 - s1 * c2 * s3;
			this->_z = c1 * c2 * s3 + s1 * s2 * c3;
			this->_w = c1 * c2 * c3 - s1 * s2 * s3;
			break;

		case EulerOrder::YXZ:
			this->_x = s1 * c2 * c3 + c1 * s2 * s3;
			this->_y = c1 * s2 * c3 - s1 * c2 * s3;
			this->_z = c1 * c2 * s3 - s1 * s2 * c3;
			this->_w = c1 * c2 * c3 + s1 * s2 * s3;
			break;

		case EulerOrder::ZXY:
			this->_x = s1 * c2 * c3 - c1 * s2 * s3;
			this->_y = c1 * s2 * c3 + s1 * c2 * s3;
			this->_z = c1 * c2 * s3 + s1 * s2 * c3;
			this->_w = c1 * c2 * c3 - s1 * s2 * s3;
			break;

		case EulerOrder::ZYX:
			this->_x = s1 * c2 * c3 - c1 * s2 * s3;
			this->_y = c1 * s2 * c3 + s1 * c2 * s3;
			this->_z = c1 * c2 * s3 - s1 * s2 * c3;
			this->_w = c1 * c2 * c3 + s1 * s2 * s3;
			break;

		case EulerOrder::YZX:
			this->_x = s1 * c2 * c3 + c1 * s2 * s3;
			this->_y = c1 * s2 * c3 + s1 * c2 * s3;
			this->_z = c1 * c2 * s3 - s1 * s2 * c3;
			this->_w = c1 * c2 * c3 - s1 * s2 * s3;
			break;

		case EulerOrder::XZY:
			this->_x = s1 * c2 * c3 - c1 * s2 * s3;
			this->_y = c1 * s2 * c3 - s1 * c2 * s3;
			this->_z = c1 * c2 * s3 + s1 * s2 * c3;
			this->_w = c1 * c2 * c3 + s1 * s2 * s3;
			break;
		}
	
		return *this;

	}

	template<typename TVecFloat, typename TAngle>
	Quaternion& setFromAxisAngle(cv::Vec<TVecFloat, 3> axis, TAngle angle) {

		// http://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm

		// assumes axis is normalized

		auto halfAngle = angle / 2;
		auto s = std::sin(halfAngle);

		_x = axis[0] * s;
		_y = axis[1] * s;
		_z = axis[2] * s;
		_w = std::cos(halfAngle);

		return *this;

	}

	template<typename TMatFloat>
	Quaternion& setFromRotationMatrix(const cv::Mat& R) {

		// http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm

		// assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)

		auto m11 = R.at<TMatFloat>(0, 0);
		auto m12 = R.at<TMatFloat>(1, 0);
		auto m13 = R.at<TMatFloat>(2, 0);
		auto m21 = R.at<TMatFloat>(0, 1);
		auto m22 = R.at<TMatFloat>(1, 1);
		auto m23 = R.at<TMatFloat>(2, 1);
		auto m31 = R.at<TMatFloat>(0, 2);
		auto m32 = R.at<TMatFloat>(1, 2);
		auto m33 = R.at<TMatFloat>(2, 2);

		auto trace = m11 + m22 + m33;

		if (trace > 0) {

			auto s = 0.5 / std::sqrt(trace + 1.0);

			_w = 0.25 / s;
			_x = (m32 - m23) * s;
			_y = (m13 - m31) * s;
			_z = (m21 - m12) * s;

		}
		else if (m11 > m22 && m11 > m33) {

			auto s = 2.0 * std::sqrt(1.0 + m11 - m22 - m33);

			_w = (m32 - m23) / s;
			_x = 0.25 * s;
			_y = (m12 + m21) / s;
			_z = (m13 + m31) / s;

		}
		else if (m22 > m33) {

			auto s = 2.0 * std::sqrt(1.0 + m22 - m11 - m33);

			_w = (m13 - m31) / s;
			_x = (m12 + m21) / s;
			_y = 0.25 * s;
			_z = (m23 + m32) / s;

		}
		else {

			auto s = 2.0 * std::sqrt(1.0 + m33 - m11 - m22);

			_w = (m21 - m12) / s;
			_x = (m13 + m31) / s;
			_y = (m23 + m32) / s;
			_z = 0.25 * s;

		}

		return *this;

	}

	template<typename TVecFloat>
	Quaternion& setFromUnitVectors(const cv::Vec<TVecFloat, 3>& vFrom, const cv::Vec<TVecFloat, 3>& vTo) {

		// assumes direction vectors vFrom and vTo are normalized

		constexpr auto EPS = 0.000001;

		auto r = vFrom.dot(vTo) + 1;

		if (r < EPS) {

			r = 0;

			if (std::abs(vFrom[0]) > std::abs(vFrom[2])) {
				_x = -vFrom[1];
				_y = vFrom[0];
				_z = 0;
				_w = r;
			}
			else {
				_x = 0;
				_y = -vFrom[2];
				_z = vFrom[1];
				_w = r;
			}

		}
		else {

			// crossVectors( vFrom, vTo ); // inlined to avoid cyclic dependency on Vector3

			_x = vFrom[1] * vTo[2] - vFrom.z * vTo[1];
			_y = vFrom[2] * vTo[0] - vFrom.x * vTo[2];
			_z = vFrom[0] * vTo[1] - vFrom.y * vTo[0];
			_w = r;

		}

		return this.normalize();
	}

	TFloat angleTo(const Quaternion& q) {

		return 2 * std::acos(std::abs(std::clamp(this->dot(q), -1, 1)));

	}

	Quaternion& rotateTowards(const Quaternion& q, TFloat step) {

		auto angle = this->angleTo(q);

		if (angle == 0) return *this;

		auto t = std::min(TFloat(1), step / angle);

		this->slerp(q, t);

		return *this;

	}

	Quaternion& identity() {

		return this->set(0, 0, 0, 1);

	}

	Quaternion& invert() {

		// quaternion is assumed to have unit length

		return this->conjugate();

	}

	Quaternion& conjugate() {

		_x *= -1;
		_y *= -1;
		_z *= -1;

		return *this;

	}

	Quaternion& dot(const Quaternion& v) {

		return _x * v._x + _y * v._y + _z * v._z + _w * v._w;

	}

	TFloat lengthSq() {

		return _x * _x + _y * _y + _z * _z + _w * _w;

	}

	TFloat length() {

		return std::sqrt(_x * _x + _y * _y + _z * _z + _w * _w);

	}

	Quaternion& normalize() {

		auto l = this->length();

		if (l == 0) {

			_x = 0;
			_y = 0;
			_z = 0;
			_w = 1;

		}
		else {

			l = TFloat(1) / l;

			_x = _x * l;
			_y = _y * l;
			_z = _z * l;
			_w = _w * l;

		}

		return *this;

	}

	Quaternion& multiply(const Quaternion& q) {
		return multiplyQuaternions(*this, q);

	}

	Quaternion& premultiply(const Quaternion& q) {
		return multiplyQuaternions(q, *this);

	}

	Quaternion& multiplyQuaternions(const Quaternion& a, const Quaternion& b) {

		// from http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/code/index.htm

		TFloat qax = a._x, qay = a._y, qaz = a._z, qaw = a._w;
		TFloat qbx = b._x, qby = b._y, qbz = b._z, qbw = b._w;

		_x = qax * qbw + qaw * qbx + qay * qbz - qaz * qby;
		_y = qay * qbw + qaw * qby + qaz * qbx - qax * qbz;
		_z = qaz * qbw + qaw * qbz + qax * qby - qay * qbx;
		_w = qaw * qbw - qax * qbx - qay * qby - qaz * qbz;

		return *this;

	}

	Quaternion& slerp(const Quaternion& qb, TFloat t) {

		if (t == 0) return *this;
		if (t == 1) return this->copy(qb);

		TFloat x = _x, y = _y, z = _z, w = _w;

		// http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/slerp/

		auto cosHalfTheta = w * qb._w + x * qb._x + y * qb._y + z * qb._z;

		if (cosHalfTheta < 0) {

			_w = -qb._w;
			_x = -qb._x;
			_y = -qb._y;
			_z = -qb._z;

			cosHalfTheta = -cosHalfTheta;

		}
		else {
			this->copy(qb);

		}

		if (cosHalfTheta >= 1.0) {

			_w = w;
			_x = x;
			_y = y;
			_z = z;

			return *this;

		}

		auto sqrSinHalfTheta = 1.0 - cosHalfTheta * cosHalfTheta;

		if (sqrSinHalfTheta <= std::numeric_limits<TFloat>::epsilon()) {

			auto s = 1 - t;
			_w = s * w + t * _w;
			_x = s * x + t * _x;
			_y = s * y + t * _y;
			_z = s * z + t * _z;

			this->normalize();
			return *this;

		}

		auto sinHalfTheta = std::sqrt(sqrSinHalfTheta);
		auto halfTheta = std::atan2(sinHalfTheta, cosHalfTheta);
		auto ratioA = std::sin((1 - t) * halfTheta) / sinHalfTheta;
		auto ratioB = std::sin(t * halfTheta) / sinHalfTheta;

		_w = (w * ratioA + _w * ratioB);
		_x = (x * ratioA + _x * ratioB);
		_y = (y * ratioA + _y * ratioB);
		_z = (z * ratioA + _z * ratioB);

		return *this;

	}

	bool equals(const Quaternion& quaternion) {

		return (quaternion._x == _x) && (quaternion._y == _y) && (quaternion._z == _z) && (quaternion._w == _w);

	}

	template<typename TArrFloat>
	Quaternion& fromArray(TArrFloat* array, int offset = 0) {

		_x = array[offset];
		_y = array[offset + 1];
		_z = array[offset + 2];
		_w = array[offset + 3];

		return *this;

	}

	cv::Mat_<TFloat> toRotationMatrix()
	{
		cv::Mat_<TFloat> R = cv::Mat_<TFloat>::eye(cv::Size(3, 3));
		auto&& m11 = R.at<TFloat>(0, 0);
		auto&& m12 = R.at<TFloat>(1, 0);
		auto&& m13 = R.at<TFloat>(2, 0);
		auto&& m21 = R.at<TFloat>(0, 1);
		auto&& m22 = R.at<TFloat>(1, 1);
		auto&& m23 = R.at<TFloat>(2, 1);
		auto&& m31 = R.at<TFloat>(0, 2);
		auto&& m32 = R.at<TFloat>(1, 2);
		auto&& m33 = R.at<TFloat>(2, 2);

		TFloat x = _x, y = _y, z = _z, w = _w;
		TFloat x2 = x + x, y2 = y + y, z2 = z + z;
		TFloat xx = x * x2, xy = x * y2, xz = x * z2;
		TFloat yy = y * y2, yz = y * z2, zz = z * z2;
		TFloat wx = w * x2, wy = w * y2, wz = w * z2;

		m11 = (1 - (yy + zz));
		m21 = (xy + wz);
		m31 = (xz - wy);

		m12 = (xy - wz);
		m22 = (1 - (xx + zz));
		m32 = (yz + wx);

		m13 = (xz + wy);
		m23 = (yz - wx);
		m33 = (1 - (xx + yy));

		return R;
	}
};

template<typename TFloat>
std::ostream& operator <<(std::ostream& s, const Quaternion<TFloat>& q)
{
	s << "[" << q._x << "," << q._y << "," << q._z << "," << q._w << "]";
	return s;
}

