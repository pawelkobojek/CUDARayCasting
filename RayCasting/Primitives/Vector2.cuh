#ifndef _VECTOR2_
#define _VECTOR2_

namespace primitives
{
	struct Vector2
	{
		float x;
		float y;

		__host__ __device__	Vector2(float x, float y);
		__host__ __device__ Vector2();
	};
}

#endif