#ifndef _RAYTRACERCPU_
#define _RAYTRACERCPU_

#include <cstdint>
#include "RayTracer.cuh"

using namespace geometries;
using namespace camera;

namespace engine
{
	class RayTracerCPU : public RayTracer
	{
	public:
		virtual void rayTrace(uint32_t* image, World* world, ICamera* camera, int width, int height);
		virtual ColorRGB shadeRay(const World* world, const Ray ray);
	};
}

#endif