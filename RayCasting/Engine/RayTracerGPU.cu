#include "RayTracerGPU.cuh"

namespace engine
{
	__device__ void RayTracerGPU::rayTrace(uint32_t* image, World* world, PinholeGPU* camera, int width, int height, int x, int y)
	{
		Vector2 pcoord = Vector2(
			((x + 0.5f) / (float)width) * 2 - 1,
			((y + 0.5f) / (float)height) * 2 - 1);
		Ray ray = camera->getRayTo(pcoord);

		shadeRayGPU(world, ray, &image[COORDS_2DTO1D(x, y)]);
	}

	__device__ void RayTracerGPU::shadeRayGPU(const World* world, const Ray ray, uint32_t* colorRgb) {

			HitInfoGPU info = world->traceRayGPU(ray);
			*colorRgb = ColorRGB(1.0f, 0.0f, 0.0f);//ColorRGB(0.0f, 0.0f, 0.0f);

			if (!info.hitObject) {
				*colorRgb = world->backgroundColor;
				return;
			}

			ColorRGB finalColor = ColorRGB(0.0f, 0.0f, 0.0f); // <==> ColorRGB::black
			PhongGPU material = info.hitObject->material;

			if(!world->gpuAnyObstacleBetween(info.hitPoint, info.hitPoint)) {
				finalColor += material.radiance(world->gpulights[0], info);
			}

			*colorRgb = finalColor;
	}
}