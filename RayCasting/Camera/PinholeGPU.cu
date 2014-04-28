#include "PinHoleGPU.cuh"

namespace camera
{
	__host__ __device__ Ray PinholeGPU::getRayTo(Vector2 pictureLocation) const
	{
		return Ray(origin, rayDirection(pictureLocation));
	}

	__host__ __device__ Vector3 PinholeGPU::rayDirection(Vector2 v) const
	{
		return onb * Vector3(v.x, v.y, -distance);
	}

	__host__ __device__ PinholeGPU::PinholeGPU(Vector3 origin, Vector3 lookAt, Vector3 up, float distance)
		: onb(origin, lookAt, up), origin(origin), distance(distance) {}

	__host__ __device__ PinholeGPU::PinholeGPU() {}

	PinholeGPU* PinholeGPU::GpuAlloc(Vector3 origin, Vector3 lookAt, Vector3 up, float distance) {
		PinholeGPU obj = PinholeGPU(origin, lookAt, up, distance);
		PinholeGPU* d_ptr;

		cudaMalloc((void**)&d_ptr, sizeof(PinholeGPU));
		cudaMemcpy(d_ptr, &obj, sizeof(PinholeGPU), cudaMemcpyHostToDevice);

		return d_ptr;
	}
}