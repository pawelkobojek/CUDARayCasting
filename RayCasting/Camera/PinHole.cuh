#ifndef _PINHOLE_
#define _PINHOLE_

#include "ICamera.cuh"


using namespace primitives;

namespace camera
{
	class Pinhole : public ICamera
	{
	private:
		OrthonormalBasis onb;
		Vector3 origin;
		float distance;

	public:
		Pinhole(Vector3 origin, Vector3 lookAt, Vector3 up, float distance);

		virtual Ray getRayTo(Vector2 pictureLocation) const;
		virtual Vector3 rayDirection(Vector2 relativeLocation) const;
	};
}

#endif