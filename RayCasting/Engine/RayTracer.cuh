#ifndef _RAYTRACER_
#define _RAYTRACER_

#include <cstdint>
#include "../Geometries/World.cuh"
#include "../Camera/OrthogonalCamera.cuh"
#include "../base.cuh"

using namespace geometries;
using namespace camera;

namespace 
{
	class RayTracer
	{
	public:
		virtual ~RayTracer() {}
		virtual void rayTrace(uint32_t* image, World* world, ICamera* camera, int width, int height) = 0;
	};
}

#endif