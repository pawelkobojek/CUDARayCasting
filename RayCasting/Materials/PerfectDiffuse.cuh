#ifndef _PERFECTDIFFUSE_
#define _PERFECTDIFFUSE_

#include "IMaterial.cuh"
#include "../Light/PointLight.cuh"
#include "../Common/ColorRGB.cuh"
#include "../Common/HitInfo.cuh"

using namespace light;

namespace materials
{
	class PerfectDiffuse : public IMaterial
	{
	private:
		ColorRGB materialColor;
	public:
		PerfectDiffuse(ColorRGB materialColor);

		virtual ColorRGB radiance(const PointLight light, const HitInfo hit) const;
	};
}

#endif