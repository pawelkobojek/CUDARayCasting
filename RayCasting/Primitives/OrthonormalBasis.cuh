#ifndef _ORTHONORMALBASIS_
#define _ORTHONORMALBASIS_

#include "Vector3.cuh"

namespace primitives
{
	class OrthonormalBasis
	{
	private:
		Vector3 u;
		Vector3 v;
		Vector3 w;

	public:
		__host__ __device__ OrthonormalBasis(Vector3 eye, Vector3 lookAt, Vector3 up);
		__host__ __device__ OrthonormalBasis();

		__host__ __device__ friend Vector3 operator*(const OrthonormalBasis onb, const Vector3 v);
	};
}

#endif