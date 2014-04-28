#include "PinHole.cuh"

namespace camera
{
	Ray Pinhole::getRayTo(Vector2 pictureLocation) const
	{
		return Ray(origin, rayDirection(pictureLocation));
	}

	Vector3 Pinhole::rayDirection(Vector2 v) const
	{
		return onb * Vector3(v.x, v.y, -distance);
	}

	Pinhole::Pinhole(Vector3 origin, Vector3 lookAt, Vector3 up, float distance)
		: onb(origin, lookAt, up), origin(origin), distance(distance) {}
}