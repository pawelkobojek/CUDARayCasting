#include "GeometricObject.cuh"

namespace geometries
{
	GeometricObject::GeometricObject(IMaterial* m) : material(m) {}
}