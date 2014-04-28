#ifndef _VECTOR3_
#define _VECTOR3_
namespace primitives
{
	struct Vector3
	{
		float x;
		float y;
		float z;

		__host__ __device__ Vector3(float x, float y, float z);
		__host__ __device__ Vector3();

		__host__ __device__ float dot(const Vector3 vector) const;
		__host__ __device__ float length() const;
		__host__ __device__ float lengthSq() const;
		__host__ __device__ Vector3 operator-() const;
		__host__ __device__ Vector3 normalized() const;
		__host__ __device__ friend Vector3 operator+(const Vector3 first, const Vector3 second);
		__host__ __device__ friend Vector3 operator-(const Vector3 first, const Vector3 second);
		__host__ __device__ friend Vector3 operator*(const Vector3 vector, const float scalar);
		__host__ __device__ friend Vector3 operator/(const Vector3 vector, const float scalar);

		__host__ __device__ friend float dot(const Vector3 first, const Vector3 second);
		__host__ __device__ friend Vector3 cross(const Vector3 first, const Vector3 second);
		__host__ __device__ friend Vector3 reflect(const Vector3 vec, const Vector3 normal);
	};
}

#endif