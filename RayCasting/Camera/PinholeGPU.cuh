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
		__host__ __device__ PinholeGPU();
		__host__ __device__ PinholeGPU(Vector3 origin, Vector3 lookAt, Vector3 up, float distance);

		__host__ __device__ Ray getRayTo(Vector2 pictureLocation) const;
		__host__ __device__ Vector3 rayDirection(Vector2 relativeLocation) const;

		static PinholeGPU* GpuAlloc(Vector3 origin, Vector3 lookAt, Vector3 up, float distance);

	};
}

#endif