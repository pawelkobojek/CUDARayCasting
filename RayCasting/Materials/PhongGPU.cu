#include "PhongGPU.cuh"
#include <cmath>

namespace materials
{
	PhongGPU::PhongGPU(ColorRGB materialColor, float diffuse, float specular, float specularExponent)
		: materialColor(materialColor), diffuse(diffuse), specular(specular), specularExponent(specularExponent) {}

	PhongGPU::PhongGPU() : materialColor(ColorRGB(0.0f, 0.0f, 0.0f)) {}

	PhongGPU::PhongGPU(Phong material) : materialColor(material.materialColor), diffuse(material.diffuse), specular(material.specular),
		specularExponent(material.specularExponent) {
	}

	__device__ ColorRGB PhongGPU::radiance(const PointLight l, const HitInfoGPU hit) const
	{
		Vector3 inDirection = (l.getPosition() - hit.hitPoint).normalized();
		float diffFactor = inDirection.dot(hit.normal);

		//	if (diffFactor < 0) return ColorRGB(0.0f, 0.0f, 0.0f);//return ColorRGB::black;

		ColorRGB result = ColorRGB(1.0f, 1.0f, 1.0f) * materialColor * diffFactor * diffuse;
		//ColorRGB result = l.getColor() * materialColor * diffFactor * diffuse;

		float pFactor = phongFactor(inDirection, hit.normal, -hit.ray.direction);
		if (pFactor != 0.f)
			result += materialColor * specular * pFactor;
		return result;
	}

	__device__ float PhongGPU::phongFactor(const Vector3 inDirection, const Vector3 normal, const Vector3 toCameraDirection) const
	{
		Vector3 reflected = reflect(inDirection, normal);
		float cosAngle = reflected.dot(toCameraDirection);
		if (cosAngle <= 0) return 0.f;

		return powf(cosAngle, specularExponent);
	}
}