#ifndef _PINHOLEGPU_
#define _PINHOLEGPU_

#include "ICamera.cuh"


using namespace primitives;

namespace camera
{
	class PinholeGPU
	{
	private:
		OrthonormalBasis onb;
		Vector3 origin;
		float distance;

	public:
		__device__ PinholeGPU();
		__device__ PinholeGPU(Vector3 origin, Vector3 lookAt, Vector3 up, float distance);

		__device__ Ray getRayTo(Vector2 pictureLocation) const;
		__device__ Vector3 rayDirection(Vector2 relativeLocation) const;

		static PinholeGPU* GpuAlloc(Vector3 origin, Vector3 lookAt, Vector3 up, float distance);

	};
}

#endif