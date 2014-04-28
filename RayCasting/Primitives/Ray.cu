#include "Ray.cuh"
#include <cfloat>

namespace primitives
{
	const float Ray::epsilon = 0.0001;

	const float Ray::huge = FLT_MAX;

	__host__ __device__ Ray::Ray() {

	}

	__host__ __device__ Ray::Ray(const Vector3 origin, const Vector3 direction)
	{
		this->origin = origin;
		this->direction = direction.normalized();
	}
}