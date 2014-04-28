#ifndef _HITINFO_
#define _HITINFO_

namespace geometries
{
	class GeometricObject;
	class World;
	class SphereGPU;
	class PlaneGPU;
}

#include "../Primitives/primitives.cuh"

using geometries::GeometricObject;
using geometries::World;
using geometries::SphereGPU;
using geometries::PlaneGPU;
using primitives::Vector3;
using primitives::Ray;

struct HitInfo
{
	GeometricObject* hitObject;
	World* world;
	Vector3 normal;
	Vector3 hitPoint;
	Ray ray;

	HitInfo();
};

struct HitInfoGPU
{
	SphereGPU* hitObject;
	PlaneGPU* hitPlane;
	World* world;
	Vector3 normal;
	Vector3 hitPoint;
	Ray ray;

	__host__ __device__ HitInfoGPU();
};

#endif