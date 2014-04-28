#ifndef _ICAMERA_
#define _ICAMERA_

#include "../Primitives/primitives.cuh"

using primitives::Ray;
using primitives::Vector2;

namespace camera
{
	class ICamera
	{
	public:
		virtual ~ICamera() {}
		virtual Ray getRayTo(Vector2 relativeLocation) const = 0;
	};
}
#endif