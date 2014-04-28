#ifndef _GEOMETRICOBJECT_
#define _GEOMETRICOBJECT_
#include "../Primitives/Ray.cuh"
#include "../Materials/IMaterial.cuh"

using primitives::Ray;
using materials::IMaterial;

namespace geometries
{
	class GeometricObject
	{
	public:
		IMaterial* material;

		GeometricObject(IMaterial* material);

		virtual ~GeometricObject() {};

		virtual bool hitTest(const Ray ray, float* distance, Vector3* normal) const = 0;
	};
}

#endif