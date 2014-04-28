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
		__device__ void shadeRayGPU(const World* world, const Ray ray, uint32_t* colorRgb);
		__device__ void rayTrace(uint32_t* image, World* world, PinholeGPU* camera, int width, int height, int x, int y);
		ColorRGB shadeRay(const World* world, const Ray ray);

		static RayTracerGPU* GPUAllocate();
	};
}
#endif