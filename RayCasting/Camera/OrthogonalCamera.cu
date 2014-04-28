#include "OrthogonalCamera.cuh"
#include <cmath>

using namespace primitives;

namespace camera
{
	OrthogonalCamera::OrthogonalCamera(Vector3 eye, float angle, Vector2 size)
		: eyePosition(eye), angle(angle), cameraSize(size)
	{
	}

	Ray OrthogonalCamera::getRayTo(Vector2 pL) const
	{
		Vector3 direction = Vector3(sinf(angle), 0, cosf(angle)).normalized();
		Vector2 offsetFromCenter = Vector2(pL.x*cameraSize.x, pL.y * cameraSize.y);
		Vector3 position = Vector3(eyePosition.x + offsetFromCenter.x * cosf(angle),
			eyePosition.y + offsetFromCenter.y,
			eyePosition.z + offsetFromCenter.x * sinf(angle));
		return Ray(position, direction);
	}
}