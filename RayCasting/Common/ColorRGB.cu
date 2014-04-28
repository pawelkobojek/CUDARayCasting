#include "ColorRGB.cuh"
#include "../base.cuh"

const ColorRGB ColorRGB::white = ColorRGB(1.f, 1.f, 1.f);
const ColorRGB ColorRGB::black = ColorRGB(0.f, 0.f, 0.f);
const ColorRGB ColorRGB::red = ColorRGB(1.f, 0.f, 0.f);
const ColorRGB ColorRGB::green = ColorRGB(0.f, 1.f, 0.f);
const ColorRGB ColorRGB::blue = ColorRGB(0.f, 0.f, 1.f);
const ColorRGB ColorRGB::gray = ColorRGB(0.5f, 0.5f, 0.5f);

__host__ __device__ ColorRGB::ColorRGB(float r, float g, float b)
	: r(r), g(g), b(b) {}

__host__ __device__
	ColorRGB::ColorRGB(const uint32_t color)
	: r((float)R(color) / 255.f), g((float)G(color) / 255.f), b((float)B(color) / 255.f) {}


__host__ __device__ ColorRGB::operator uint32_t() const
{
	ColorRGB color = this->correct();
	return COLOR((int)(color.r*255.f), (int)(color.g*255.f), (int)(color.b*255.f), 255);
}

__host__ __device__ ColorRGB ColorRGB::correct() const
{
	ColorRGB color = *this;
	color.r = (color.r < 0) ? 0.f : (color.r>1) ? 1 : color.r;
	color.g = (color.g < 0) ? 0.f : (color.g>1) ? 1 : color.g;
	color.b = (color.b < 0) ? 0.f : (color.b>1) ? 1 : color.b;
	return color;
}

__host__ __device__ ColorRGB operator+(const ColorRGB first, const ColorRGB second)
{
	return ColorRGB(first.r + second.r, first.g + second.g, first.b + second.b);
}

__host__ __device__
	ColorRGB operator+=(ColorRGB& first, const ColorRGB second)
{
	first = first + second;
	return first;
}

__host__ __device__ ColorRGB operator*(const ColorRGB c, const float v)
{
	return ColorRGB(c.r*v, c.g*v, c.b*v);
}
__host__ __device__ ColorRGB operator*(const ColorRGB f, const ColorRGB s)
{
	return ColorRGB(f.r*s.r, f.g*s.g, f.b*s.b);
}
__host__ __device__ ColorRGB operator/(const ColorRGB c, const float v)
{
	return c * (1 / v);
}