#include "Plane.cuh"

namespace geometries
{
	Plane::Plane(const Vector3 point, const Vector3 normal, IMaterial* material)
		: GeometricObject(material), point(point), normal(normal)
	{
	}

	bool Plane::hitTest(const Ray ray, float* distance, Vector3* outNormal) const
	{
		float t = (point - ray.origin).dot(normal) / ray.direction.dot(normal);

		if (t > ray.epsilon)
		{
			*distance = t;
			*outNormal = normal;
			return true;
		}

		return false;
	}
}