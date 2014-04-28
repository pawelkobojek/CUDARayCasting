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
		int currentObject;
		int currentLight;
	public:
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
		__host__ __device__	HitInfo traceRay(primitives::Ray ray) const;
		__device__ HitInfoGPU traceRayGPU(primitives::Ray ray) const;
		bool add(GeometricObject* object);
		bool addLight(PointLight* light);
		__host__ __device__	bool anyObstacleBetween(Vector3 pointA, Vector3 pointB) const;
		__host__ __device__ bool gpuAnyObstacleBetween(Vector3 pointA, Vector3 pointB) const;

		bool gpuAdd(SphereGPU* object);
		bool gpuAddLight(PointLight* light);

		static World* GpuAlloc(uint32_t backgroundColor, int objectsCount, int lights);
		static World* GpuAllocFrom(World* world);
		static void GpuFree(World* d_ptr);
	};
}

#endif