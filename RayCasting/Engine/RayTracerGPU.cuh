#ifndef _RAYTRACERGPU_
#define _RAYTRACERGPU_

#include "../base.cuh"
#include <cstdint>
#include "../Geometries/World.cuh"
#include "../Camera/PinholeGPU.cuh"
#include "../base.cuh"

using namespace geometries;
using namespace camera;

namespace engine
{
	class RayTracerGPU
	{
	public:
		__device__ void shadeRayGPU(SphereGPU* world, int count, const Ray ray, uint32_t* colorRgb);
		__device__ void rayTrace(uint32_t* image, SphereGPU* world, int count, PinholeGPU* camera, int width, int height, int x, int y);
		__device__ HitInfoGPU traceRayGPU(primitives::Ray ray, SphereGPU* gpushperes, int count) const;
		__device__ bool gpuAnyObstacleBetween(SphereGPU* gpuspheres, int count, Vector3 pointA, Vector3 pointB);
	};
}
#endif