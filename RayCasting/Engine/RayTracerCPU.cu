#include "RayTracerCPU.cuh"

namespace engine
{
	void RayTracerCPU::rayTrace(uint32_t* image, World* world, ICamera* camera, int width, int height)
	{
		for (int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
		{
			Vector2 pcoord = Vector2(
				((x + 0.5f) / (float)width) * 2 - 1,
				((y + 0.5f) / (float)height) * 2 - 1);
			Ray ray = camera->getRayTo(pcoord);

			image[COORDS_2DTO1D(x, y)] = shadeRay(world, ray);
		}
	}

	ColorRGB RayTracerCPU::shadeRay(const World* world, const Ray ray)
	{
		HitInfo info = world->traceRay(ray);

		if (!info.hitObject) return world->backgroundColor;

		ColorRGB finalColor = ColorRGB::black;
		IMaterial* material = info.hitObject->material;

		for (int i = 0; i < world->lightsCount; i++)
		if (world->lights[i] && !world->anyObstacleBetween(info.hitPoint, world->lights[i]->getPosition()))
			finalColor += material->radiance(*(world->lights[i]), info);
		return finalColor;
	}
}