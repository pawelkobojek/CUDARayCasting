#ifndef _PLANEGPU_
#define _PLANEGPU_

#include "../Primitives/primitives.cuh"
#include "../Materials/Phong.cuh"
#include "../Materials/PhongGPU.cuh"

using namespace primitives;
using primitives::Vector3;
using materials::Phong;
using materials::PhongGPU;
using primitives::Ray;

namespace geometries
{
	class PlaneGPU
	{
	public:
		Vector3 point;
		Vector3 normal;
		PhongGPU material;
		PlaneGPU(const Vector3 point, const Vector3 normal, Phong material);
		PlaneGPU();
		__device__ bool hitTest(const Ray ray, float* distance, Vector3* normal) const;
	};
}

#endif