#include "OrthonormalBasis.cuh"

namespace primitives
{
	__host__ __device__ OrthonormalBasis::OrthonormalBasis(Vector3 eye, Vector3 lookAt, Vector3 up)
	{
		w = (eye - lookAt).normalized();
		u = cross(up, w).normalized();
		v = cross(w, u);
	}

	__host__ __device__ OrthonormalBasis::OrthonormalBasis() {}

	__host__ __device__ Vector3 operator*(const OrthonormalBasis onb, const Vector3 v)
	{
		return (onb.u * v.x + onb.v * v.y + onb.w * v.z);
	}
}