#include "Sphere.cuh"
#include <cmath>

namespace geometries
{
	Sphere::Sphere(const Vector3 center, const float radius, IMaterial* material)
		: GeometricObject(material), center(center), radius(radius) {}

	bool Sphere::hitTest(const Ray ray, float* minDistance, Vector3* outNormal) const
	{
		float t;
		Vector3 distance = ray.origin - center;
		float a = ray.direction.lengthSq();
		float b = (distance * 2).dot(ray.direction);
		float c = distance.lengthSq() - radius * radius;
		float disc = b*b - 4 * a*c;

		if (disc < 0) return false;
		float discSq = sqrtf(disc);
		float denom = 2 * a;

		t = (-b - discSq) / denom;
		if (t < Ray::epsilon)
		{
			t = (-b + discSq) / denom;
		}
		if (t < Ray::epsilon)
		{
			return false;
		}

		Vector3 hitPoint = (ray.origin + ray.direction * t);
		*outNormal = (hitPoint - center).normalized();
		*minDistance = t;
		return true;
	}
}