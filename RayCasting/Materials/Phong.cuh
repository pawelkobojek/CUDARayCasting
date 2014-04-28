#ifndef _PHONG_
#define _PHONG_

#include "IMaterial.cuh"

namespace materials
{
	class Phong : public IMaterial
	{
	public:
		ColorRGB materialColor;
		float diffuse;
		float specular;
		float specularExponent;

		Phong(ColorRGB materialColor, float diffuse, float specular, float specularExponent);
		Phong();

		virtual ColorRGB radiance(const PointLight light, const HitInfo hit) const;

		virtual float phongFactor(const Vector3 inDirection, const Vector3 normal, const Vector3 toCameraDirection) const;
	};
}

#endif