#include "PlaneGPU.cuh"

namespace geometries
{
	PlaneGPU::PlaneGPU(const Vector3 point, const Vector3 normal, Phong material)
		: material(material), normal(normal), point(point)
	{
	}

	PlaneGPU::PlaneGPU(){}

	__device__ bool PlaneGPU::hitTest(const Ray ray, float* distance, Vector3* outNormal) const
	{
		float t = (point - ray.origin).dot(normal) / ray.direction.dot(normal);

		if (t > 0)//ray.epsilon)
		{
			*distance = t;
			*outNormal = normal;
			return true;
		}

		return false;
	}
}