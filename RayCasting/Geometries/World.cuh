#ifndef _WORLD_
#define _WORLD_

#include "GeometricObject.cuh"
#include "../Common/HitInfo.cuh"
#include "../Primitives/Ray.cuh"
#include "../Light/PointLight.cuh"
#include "../Geometries/SphereGPU.cuh"
#include "../Geometries/PlaneGPU.cuh"

using namespace light;

namespace geometries
{
	class World
	{
	public:
		int currentObject;
		int currentLight;
		uint32_t backgroundColor;
		int objectsCount;
		GeometricObject** objects;
		int lightsCount;
		PointLight** lights;

		SphereGPU* gpuspheres;
		PointLight* gpulights;
		PlaneGPU* gpuplane;

		World(uint32_t backgroundColor, int objectsCount, int lightsCount);
		__host__ __device__ ~World();
		HitInfo traceRay(primitives::Ray ray) const;
		__device__ HitInfoGPU traceRayGPU(primitives::Ray ray) const;
		bool add(GeometricObject* object);
		bool addLight(PointLight* light);
		bool anyObstacleBetween(Vector3 pointA, Vector3 pointB) const;
		__device__ bool gpuAnyObstacleBetween(Vector3 pointA, Vector3 pointB) const;

		bool gpuAdd(SphereGPU* object);
		bool gpuAddLight(PointLight* light);

		static World* GpuAlloc(uint32_t backgroundColor, int objectsCount, int lights);
		static World* GpuAllocFrom(World* world);
		static void GpuFree(World* d_ptr);
	};
}

#endif