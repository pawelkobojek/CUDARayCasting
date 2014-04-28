#ifndef _RAY_
#define _RAY_

#include "Vector3.cuh"

namespace primitives
{
	struct Ray
	{
		static const float epsilon;
		static const float huge;

		Vector3 origin;
		Vector3 direction;

		__host__ __device__	Ray();
		__host__ __device__ Ray(const Vector3 origin, const Vector3 direction);
	};
}

#endif