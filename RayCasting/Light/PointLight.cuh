#ifndef _POINTLIGHT_
#define _POINTLIGHT_

#include "../Primitives/primitives.cuh"
#include "../Common/ColorRGB.cuh"

using namespace primitives;

namespace light
{
	class PointLight
	{
	public:
		Vector3 position;
		ColorRGB color;
		__host__ __device__ Vector3 getPosition() const;
		ColorRGB getColor() const;

		__host__ __device__ PointLight(Vector3 position, ColorRGB color);
		PointLight();
	};
}

#endif