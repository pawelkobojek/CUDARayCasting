#ifndef _IMATERIAL_
#define _IMATERIAL_

#include "../Common/ColorRGB.cuh"
#include "../Common/HitInfo.cuh"
#include "../Light/PointLight.cuh"

using namespace light;

namespace materials
{
	class IMaterial
	{
	public:
		virtual ~IMaterial() {}

		virtual ColorRGB radiance(const PointLight light, const HitInfo hit) const = 0;
	};
}

#endif