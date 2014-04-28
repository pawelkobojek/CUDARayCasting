#ifndef _PLANE_
#define _PLANE_

#include "GeometricObject.cuh"
#include "../Primitives/primitives.cuh"

using namespace primitives;

namespace geometries
{
	class Plane : public GeometricObject
	{
	public:
		Vector3 point;
		Vector3 normal;
		Plane(const Vector3 point, const Vector3 normal, IMaterial* material);
		virtual bool hitTest(const Ray ray, float* distance, Vector3* normal) const;
	};
}

#endif