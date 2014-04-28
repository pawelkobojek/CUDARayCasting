#ifndef _ORTHOGONALCAMERA_
#define _ORTHOGONALCAMERA_

#include "ICamera.cuh"
#include "../Primitives/primitives.cuh"

using namespace primitives;

namespace camera
{
	class OrthogonalCamera : public ICamera
	{
	public:
		Vector3 eyePosition;
		float angle;
		Vector2 cameraSize;

		OrthogonalCamera(Vector3 eye, float angle, Vector2 size);
		virtual Ray getRayTo(Vector2 pictureLocation) const;
	};
}
#endif