#ifndef _SPHERE_
#define _SPHERE_

#include "GeometricObject.cuh"
#include "../Primitives/primitives.cuh"
#include <cstdint>

using primitives::Vector3;

namespace geometries
{
	class Sphere : public GeometricObject
	{
	public:
		Vector3 center;
		float radius;

		Sphere(const Vector3 center, const float radius, IMaterial* material);
		virtual ~Sphere() {}
		virtual bool hitTest(const Ray ray, float* distance, Vector3* normal) const;
	};
}

#endif