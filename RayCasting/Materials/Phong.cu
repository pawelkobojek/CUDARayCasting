#include "Phong.cuh"
#include <cmath>

namespace materials
{
	Phong::Phong(ColorRGB materialColor, float diffuse, float specular, float specularExponent)
		: materialColor(materialColor), diffuse(diffuse), specular(specular), specularExponent(specularExponent) {}

	Phong::Phong() : materialColor(ColorRGB(0.0f, 0.0f, 0.0f)) {}

	__host__ __device__
		ColorRGB Phong::radiance(const PointLight l, const HitInfo hit) const
	{
			Vector3 inDirection = (l.getPosition() - hit.hitPoint).normalized();
			float diffFactor = inDirection.dot(hit.normal);

			if (diffFactor < 0) return ColorRGB(0.0f, 0.0f, 0.0f);//return ColorRGB::black;

			ColorRGB result = l.getColor() * materialColor * diffFactor *diffuse;

			float pFactor = phongFactor(inDirection, hit.normal, -hit.ray.direction);
			if (pFactor != 0.f)
				result += materialColor * specular * pFactor;
			return result;
		}

	float Phong::phongFactor(const Vector3 inDirection, const Vector3 normal, const Vector3 toCameraDirection) const
	{
		Vector3 reflected = reflect(inDirection, normal);
		float cosAngle = reflected.dot(toCameraDirection);
		if (cosAngle <= 0) return 0.f;

		return powf(cosAngle, specularExponent);
	}
}