#include "Vector2.cuh"

namespace primitives
{
	__host__ __device__ Vector2::Vector2()
	{
		x = y = 0.0f;
	}

	__host__ __device__ Vector2::Vector2(float x, float y)
	{
		this->x = x;
		this->y = y;
	}
}