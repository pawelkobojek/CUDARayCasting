#include "RayTracerGPU.cuh"

#define BACKGROUND_COLOR COLOR(200,200,255,255)

namespace engine
{
	__device__ void RayTracerGPU::rayTrace(uint32_t* image, SphereGPU* world, int count, PinholeGPU* camera, int width, int height, int x, int y)
	{
		Vector2 pcoord = Vector2(
			((x + 0.5f) / (float)width) * 2 - 1,
			((y + 0.5f) / (float)height) * 2 - 1);
		Ray ray = camera->getRayTo(pcoord);

		shadeRayGPU(world, count, ray, &image[COORDS_2DTO1D(x, y)]);
	}

	__device__ void RayTracerGPU::shadeRayGPU(SphereGPU* world, int count, const Ray ray, uint32_t* colorRgb) {

			HitInfoGPU info = traceRayGPU(ray, world, count);
			//*colorRgb = ColorRGB(1.0f, 0.0f, 0.0f);//ColorRGB(0.0f, 0.0f, 0.0f);

			if (!info.hitObject) {
				*colorRgb = BACKGROUND_COLOR;
				return;
			}

			ColorRGB finalColor = ColorRGB(0.0f, 0.0f, 0.0f); // <==> ColorRGB::black
			PhongGPU material = info.hitObject->material;

			if(!gpuAnyObstacleBetween(world, count, info.hitPoint, info.hitPoint)) {
				finalColor += material.radiance(PointLight(Vector3(0, 5, -5), ColorRGB(1.0f, 1.0f, 1.0f)), info);
			}

			*colorRgb = finalColor;
	}

	__device__ HitInfoGPU RayTracerGPU::traceRayGPU(primitives::Ray ray, SphereGPU* gpuspheres, int count) const {
		HitInfoGPU result = HitInfoGPU();
		Vector3 normal = Vector3();
		float minimalDistance = 1000000.0f;;//FLT_MAX;//Ray::huge;
		float hitDistance = 0;

		for (int i = 0; i < count; i++) {
			if (gpuspheres[i].hitTest(ray, &hitDistance, &normal) &&
				hitDistance < minimalDistance)
			{
				minimalDistance = hitDistance;
				result.hitObject = &gpuspheres[i];
				result.normal = normal;
			}
		}

		if (result.hitObject) {
			result.hitPoint = ray.origin + ray.direction * minimalDistance;
			result.ray = ray;
			result.d_world = gpuspheres;
		}

		return result;
	}

	__device__ bool RayTracerGPU::gpuAnyObstacleBetween(SphereGPU* gpuspheres, int count, Vector3 pointA, Vector3 pointB) {
		Vector3 vectorAB = pointB - pointA;
		float distAB = vectorAB.length();
		float currDistance = 1000000.0f;//FLT_MAX;//Ray::huge;

		Ray ray = Ray(pointA, vectorAB);
		Vector3 ignored = Vector3();

		for (int i = 0; i < count; i++)
			if (gpuspheres[i].hitTest(ray, &currDistance, &ignored) && currDistance < distAB)
				return true;
		//if(gpuplane->hitTest(ray, &currDistance, &ignored) && currDistance < distAB)
		//	return true;
		return false;
	}
}