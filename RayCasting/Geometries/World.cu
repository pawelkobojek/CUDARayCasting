#include "World.cuh"
#include "../Geometries/Sphere.cuh"
#include "../Geometries/Plane.cuh"
#include "../Geometries/PlaneGPU.cuh"

#include <cfloat>
#include <iostream>
#include <typeinfo>

using namespace std;

namespace geometries
{
	World::World(uint32_t backgroundColor, int objectsCount, int lightsCount)
		: backgroundColor(backgroundColor), currentLight(0), currentObject(0),
		objectsCount(objectsCount), lightsCount(lightsCount)
	{
		objects = new GeometricObject*[objectsCount];
		lights = new PointLight*[lightsCount];
		for (int i = 0; i < objectsCount; i++)
			objects[i] = 0;
		for (int i = 0; i < lightsCount; i++)
			lights[i] = 0;
	}
	__host__ __device__ World::~World()
	{
		//for (int i = 0; i < objectsCount; i++)
		//if (objects[i])
		//delete objects[i];
		//delete[] objects;
		objects = 0;
		//for (int i = 0; i < lightsCount; i++)
		//if (lights[i])
		//delete lights[i];
		//delete[] lights;
		lights = 0;
	}

	bool World::add(GeometricObject* object)
	{
		bool res = currentObject != objectsCount;
		if (res)
			objects[currentObject++] = object;
		return res;
	}

	bool World::addLight(PointLight* l)
	{
		bool res = currentLight != lightsCount;
		if (res)
			lights[currentLight++] = l;
		return res;
	}

	__device__ HitInfoGPU World::traceRayGPU(primitives::Ray ray) const {
		HitInfoGPU result = HitInfoGPU();
		Vector3 normal = Vector3();
		float minimalDistance = FLT_MAX;//Ray::huge;
		float hitDistance = 0;

		for (int i = 0; i < currentObject; i++) {
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
			result.world = (World*) this;
		}

		return result;
	}

	HitInfo World::traceRay(primitives::Ray ray) const
	{
		HitInfo result = HitInfo();
		Vector3 normal = Vector3();
		float minimalDistance = FLT_MAX;//Ray::huge;
		float hitDistance = 0;

		for (int i = 0; i < currentObject; i++)
			if (objects[i]->hitTest(ray, &hitDistance, &normal) &&
				hitDistance < minimalDistance)
			{
				minimalDistance = hitDistance;
				result.hitObject = objects[i];
				result.normal = normal;
			}

			if (result.hitObject)
			{
				result.hitPoint = ray.origin + ray.direction * minimalDistance;
				result.ray = ray;
				result.world = (World*) this;
			}
			return result;
	}

	bool World::anyObstacleBetween(Vector3 pointA, Vector3 pointB) const
	{
		Vector3 vectorAB = pointB - pointA;
		float distAB = vectorAB.length();
		float currDistance = FLT_MAX;//Ray::huge;

		Ray ray = Ray(pointA, vectorAB);
		Vector3 ignored = Vector3();

		for (int i = 0; i < currentObject; i++)
			if (objects[i]->hitTest(ray, &currDistance, &ignored) && currDistance < distAB)
				return true;
		return false;
	}

	__device__ bool World::gpuAnyObstacleBetween(Vector3 pointA, Vector3 pointB) const {
		Vector3 vectorAB = pointB - pointA;
		float distAB = vectorAB.length();
		float currDistance = FLT_MAX;//Ray::huge;

		Ray ray = Ray(pointA, vectorAB);
		Vector3 ignored = Vector3();

		for (int i = 0; i < currentObject; i++)
			if (gpuspheres[i].hitTest(ray, &currDistance, &ignored) && currDistance < distAB)
				return true;
		//if(gpuplane->hitTest(ray, &currDistance, &ignored) && currDistance < distAB)
		//	return true;
		return false;
	}

	World* World::GpuAlloc(uint32_t backgroundColor, int objectsCount, int lights) {
		World obj = World(backgroundColor, objectsCount, lights);
		World* d_ptr;

		cudaMalloc((void**) &d_ptr, sizeof(World));
		cudaMemcpy(d_ptr, &obj, sizeof(World), cudaMemcpyHostToDevice);

		return d_ptr;
	}

	World* World::GpuAllocFrom(World* world) {
		World* d_ptr;
		PlaneGPU* gpuplane;
		PlaneGPU* d_plane;
		SphereGPU* spheres = new SphereGPU[world->objectsCount];
		PointLight* lights = new PointLight[world->lightsCount];
		PointLight* d_lights;
		PointLight** d_ptr_lights;
		SphereGPU* d_spheres;
		SphereGPU** d_spheres_ptr;
		int gpuSpheresCount = 0;
		int gpuLightsCount = 0;

		for (int i = 0; i < world->objectsCount; i++)
		{
			if(world->objects[i] != NULL && typeid(*(world->objects[i])) == typeid(Sphere) ) {
				spheres[i] = SphereGPU(((Sphere*)world->objects[i])->center,
					((Sphere*)world->objects[i])->radius,
					*((Phong*)(((Sphere*)world->objects[i])->material)));
				//cout << "Added sphere" << endl;
				gpuSpheresCount++;
			} else if(world->objects[i] != NULL && typeid(*(world->objects[i])) == typeid(Plane) ) {
				gpuplane = new PlaneGPU(((Plane*)world->objects[i])->point,
					((Plane*)world->objects[i])->normal,
					*((Phong*)(((Plane*)world->objects[i])->material)));
				cout << "Added plane" << endl;
			}
		}

		for (int i = 0; i < world->lightsCount; i++)
		{
			if(world->lights[i] != NULL) {
				lights[i] = PointLight(world->lights[i]->position, world->lights[i]->color);
				gpuLightsCount++;
			}
		}
		/*
		SphereGPU** d_spheres_to_alloc = new SphereGPU*[gpuSpheresCount];
		for (int i = 0; i < gpuSpheresCount; i++)
		{
		cudaMalloc((void**)&d_spheres_to_alloc[i], sizeof(SphereGPU));
		}

		SphereGPU** d_sphere_array;
		cudaMalloc(*/

		cudaMalloc((void**) &d_ptr, sizeof(World));
		cudaMemcpy(d_ptr, world, sizeof(World), cudaMemcpyHostToDevice);
		cudaMalloc((void**) &d_spheres, sizeof(SphereGPU) * gpuSpheresCount);
		cudaMemcpy(d_spheres, spheres, sizeof(SphereGPU) * gpuSpheresCount, cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_ptr->gpuspheres), &d_spheres, sizeof(SphereGPU*), cudaMemcpyHostToDevice);

		if(gpuplane != NULL) {
			cudaMalloc((void**) &d_plane, sizeof(PlaneGPU));
			cudaMemcpy(d_plane, gpuplane, sizeof(PlaneGPU), cudaMemcpyHostToDevice);
			cudaMemcpy(&(d_ptr->gpuplane), &d_plane, sizeof(int*), cudaMemcpyHostToDevice);
		}

		cudaMalloc((void**) &d_lights, sizeof(PointLight) * gpuLightsCount);
		cudaMemcpy(d_lights, lights, sizeof(PointLight) * gpuLightsCount, cudaMemcpyHostToDevice);
		cudaMemcpy(&(d_ptr->gpulights), &d_lights, sizeof(PointLight*), cudaMemcpyHostToDevice);

		return d_ptr;
	}

	void World::GpuFree(World* d_ptr) {
		cudaFree(d_ptr->gpuplane);
		cudaFree(d_ptr->gpulights);
		cudaFree(d_ptr->gpuspheres);
		cudaFree(d_ptr);
	}

	bool World::gpuAdd(SphereGPU* object) {
		return true;
	}
}