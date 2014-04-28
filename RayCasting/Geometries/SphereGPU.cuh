#ifndef _SPHEREGPU_
#define _SPHEREGPU_

#include "../Primitives/primitives.cuh"
#include "../Materials/Phong.cuh"
#include "../Materials/PhongGPU.cuh"

#include <cstdint>

using primitives::Vector3;
using materials::Phong;
using materials::PhongGPU;
using primitives::Ray;

namespace geometries
{
	class SphereGPU
	{
	public:
		Vector3 center;
		float radius;
		PhongGPU material;

		__device__ SphereGPU();
		__device__ SphereGPU(const Vector3 center, const float radius, Phong material);
		__device__ ~SphereGPU() {}
		__device__ bool hitTest(const Ray ray, float* distance, Vector3* normal) const;
	};
}

#endif