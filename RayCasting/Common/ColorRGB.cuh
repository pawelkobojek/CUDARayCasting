#ifndef _COLORRGB_
#define _COLORRGB_

#include <cstdint>

struct ColorRGB
{
public:
	float r;
	float g;
	float b;

	__host__ __device__ ColorRGB(float r, float g, float b);
	__host__ __device__ ColorRGB(const uint32_t color);


	__host__ __device__ operator uint32_t() const;
	__host__ __device__ ColorRGB correct() const;
	friend ColorRGB operator+(const ColorRGB first, const ColorRGB second);
	__host__ __device__ friend ColorRGB operator+=(ColorRGB& first, const ColorRGB second);
	__host__ __device__ friend ColorRGB operator*(const ColorRGB color, const float val);
	__host__ __device__ friend ColorRGB operator*(const ColorRGB first, const ColorRGB second);
	__host__ __device__ friend ColorRGB operator/(const ColorRGB color, const float val);

	static const ColorRGB white;
	static const ColorRGB black;
	static const ColorRGB red;
	static const ColorRGB green;
	static const ColorRGB blue;
	static const ColorRGB gray;
};

#endif