#ifndef _PHONGGPU_
#define _PHONGGPU_

#include "../Materials/Phong.cuh"

struct HitInfoGPU;

namespace materials
{
	class PhongGPU
	{
	public:
		ColorRGB materialColor;
		float diffuse;
		float specular;
		float specularExponent;

		PhongGPU(Phong material);
		PhongGPU(ColorRGB materialColor, float diffuse, float specular, float specularExponent);
		PhongGPU();

		__device__ ColorRGB radiance(const PointLight light, const HitInfoGPU hit) const;

		__device__ float phongFactor(const Vector3 inDirection, const Vector3 normal, const Vector3 toCameraDirection) const;
	};
}

#endif