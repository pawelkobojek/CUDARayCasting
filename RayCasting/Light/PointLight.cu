#include "PointLight.cuh"

namespace light
{
	PointLight::PointLight(Vector3 position, ColorRGB color)
		: position(position), color(color) {}
	PointLight::PointLight() : color(ColorRGB(1.0f, 0.0f, 0.0f)) {
	}
	__host__ __device__
		Vector3 PointLight::getPosition() const
	{
		return position;
	}

	ColorRGB PointLight::getColor() const
	{
		return color;
	}
}