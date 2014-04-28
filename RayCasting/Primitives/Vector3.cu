#include "Vector3.cuh"
#include <cmath>

namespace primitives
{
	__host__ __device__ Vector3::Vector3()
	{
		this->x = this->y = this->z = 0;
	}

	__host__ __device__ Vector3::Vector3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	__host__ __device__ Vector3 operator+(const Vector3 first, const Vector3 second)
	{
		return Vector3(first.x + second.x, first.y + second.y, first.z + second.z);
	}

	__host__ __device__ Vector3 operator-(const Vector3 first, const Vector3 second)
	{
		return Vector3(first.x - second.x, first.y - second.y, first.z - second.z);
	}

	__host__ __device__ Vector3 operator*(const Vector3 vector, const float scalar)
	{
		return Vector3(vector.x*scalar, vector.y*scalar, vector.z*scalar);
	}

	__host__ __device__ Vector3 operator/(const Vector3 vector, const float scalar)
	{
		return Vector3(vector.x / scalar, vector.y / scalar, vector.z / scalar);
	}

	__host__ __device__ float Vector3::dot(const Vector3 vector) const
	{
		return x*vector.x + y*vector.y + z*vector.z;
	}

	__host__ __device__ float dot(const Vector3 first, const Vector3 second)
	{
		return first.dot(second);
	}

	__host__ __device__ Vector3 cross(const Vector3 first, const Vector3 second)
	{
		return Vector3(first.y*second.z - first.z * second.y,
			first.z*second.x - first.x*second.z,
			first.x*second.y - first.y*second.x);
	}

	__host__ __device__ Vector3 reflect(const Vector3 vec, const Vector3 normal)
	{
		float dot = normal.dot(vec);
		return normal * dot * 2 - vec;
	}

	__host__ __device__ float Vector3::length() const
	{
		return sqrtf(x*x + y*y + z*z);
	}

	__host__ __device__ float Vector3::lengthSq() const
	{
		return x*x + y*y + z*z;
	}

	__host__ __device__ Vector3 Vector3::normalized() const
	{
		return Vector3(*this) / length();
	}

	__host__ __device__ Vector3 Vector3::operator-() const
	{
		return Vector3(-x, -y, -z);
	}
}