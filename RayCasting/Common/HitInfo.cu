#include "HitInfo.cuh"
#include "../Primitives/primitives.cuh"
#include "../Geometries/GeometricObject.cuh"

HitInfo::HitInfo()
	: hitObject(0), world(0), normal(), hitPoint(), ray(Vector3(0, 0, 0), Vector3(0, 0, 0)) {}

__host__ __device__ HitInfoGPU::HitInfoGPU()
	: hitObject(0), world(0), normal(), hitPoint(), ray(Vector3(0, 0, 0), Vector3(0, 0, 0)) {}