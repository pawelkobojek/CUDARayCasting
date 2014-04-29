#include <SDL.h>
#include <cstdlib>
#include <iostream>

#include "MainWindow.cuh"
//#include "RayCastingGPU.h"
//#include "RayCastingCPU.h"
#include "Engine/RayTracerCPU.cuh"
#include "Engine/RayTracerGPU.cuh"
#include "Geometries/Sphere.cuh"
#include "Geometries/SphereGPU.cuh"
#include "Geometries/Plane.cuh"
#include "Camera/PinHole.cuh"
#include "Camera/PinholeGPU.cuh"
#include "Common/ColorRGB.cuh"
#include "Materials/IMaterial.cuh"
#include "Materials/Phong.cuh"
#include "Materials/PerfectDiffuse.cuh"

#define BACKGROUND_COLOR COLOR(200,200,255,255)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace std;
using namespace engine;
using namespace geometries;
using namespace camera;
using namespace materials;

__global__ void rayTraceKernelTest(uint32_t* image, int width, int height) {
	//int X = threadIdx.x;// / WIDTH;
	//int Y = threadIdx.y;// %  HEIGHT;

	//for(int X = 0; X < width; X++) {
	//for (int Y = 0; Y < height; Y++)
	//{
	//image[PIXEL(X, Y)] = 0xFF0000;//ColorRGB(1.0f, 0.0f, 0.0f);
	//}
	//}
}

__global__ void rayTraceKernel(RayTracerGPU rayTracerGPU, uint32_t* image, World* world, int count, PinholeGPU* camera, int width, int height) {
	extern __shared__ SphereGPU d_world[];

	if(threadIdx.x < world->objectsCount) {
		d_world[threadIdx.x] = world->gpuspheres[threadIdx.x];
		//for(int i = 0; i < world->objectsCount; ++i) {
			//d_world[i] = world->gpuspheres[i];
		//}
	}
	__syncthreads();
	rayTracerGPU.rayTrace(image, d_world, count, camera, width, height, blockIdx.x, threadIdx.x);
}

int main(int argc, char** argv)
{
	MainWindow window = MainWindow();
	uint32_t* data = new uint32_t[TEXTURE_SIZE];
	uint32_t* d_data;
	RayTracerCPU rayTracerCPU;
	RayTracerGPU rayTracerGPU;
	srand(time(NULL));

	SphereGPU* spheres = new SphereGPU[3];
	spheres[0] = SphereGPU(Vector3(-4.0f, 0, 0), 2, Phong(ColorRGB::red, 0.5f, 1.f, 30.f));
	spheres[1] = SphereGPU(Vector3(4.0f, 0, 0), 2, Phong(ColorRGB::red, 0.5f, 1.f, 30.f));
	spheres[2] = SphereGPU(Vector3(0, 0, 3.0f), 2, Phong(ColorRGB::red, 0.5f, 1.f, 30.f));


	IMaterial* redMat = new Phong(ColorRGB::red, 0.5f, 1.f, 30.f);
	IMaterial* greenMat = new Phong(ColorRGB::green, 0.5f, 1.f, 30.f);
	IMaterial* blueMat = new Phong(ColorRGB::blue, 0.5f, 1.f, 30.f);
	IMaterial* grayMat = new Phong(ColorRGB::gray, 0.5f, 1.f, 30.f);

	World world = World(BACKGROUND_COLOR, 400, 5);
	world.add(new Sphere(Vector3(-4.0f, 0, 0), 2, redMat));
	world.add(new Sphere(Vector3(4.0f, 0, 0), 2, greenMat));
	world.add(new Sphere(Vector3(0, 0, 3.0f), 2, blueMat));
	world.add(new Plane(Vector3(0, -2, 0), Vector3(0, 1, 0), grayMat));

	for(float i = -200.0f; i < 199.0f; i+=1.0f) {
			world.add(new Sphere(Vector3(i, 0, 0), 1, redMat));
	}

	world.addLight(new PointLight(Vector3(0, 5, -5), ColorRGB::white));

	ICamera* cam = new Pinhole(Vector3(0, 1, -8), Vector3(0, 0, 0), Vector3(0, -1, 0), 1);
	PinholeGPU pincam = PinholeGPU(Vector3(0, 1, -8), Vector3(0, 0, 0), Vector3(0, -1, 0), 1);
	RayTracerGPU rtGpu = RayTracerGPU();
	RayTracerGPU* d_rtGpu;
	PinholeGPU* d_cam = PinholeGPU::GpuAlloc(Vector3(0, 1, -8), Vector3(0, 0, 0), Vector3(0, -1, 0), 1);
	cudaMalloc((void**)&d_data, TEXTURE_SIZE * sizeof(uint32_t));
	cudaMemcpy(d_data, data, TEXTURE_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_rtGpu, sizeof(RayTracerGPU));
	cudaMemcpy(d_rtGpu, &rtGpu, sizeof(RayTracerGPU), cudaMemcpyHostToDevice);

	cout << "RayTracerGPU: " << sizeof(RayTracerGPU) << "\nuint32_t: " << sizeof(uint32_t*) 
		<< "\nWorld: " << sizeof(World) << "\nPinholeGPU: " << sizeof(PinholeGPU) << "\nTotal size of parameteres: "
		<< sizeof(RayTracerGPU) + sizeof(uint32_t*) + sizeof(World) + sizeof(PinholeGPU) << endl;

	World* d_world_ptr = World::GpuAllocFrom(&world);

	bool quit = false;
	bool computing = true;
	//while (!quit)
	{
		window.fpsLimiter.startFrame();

		while (SDL_PollEvent(&window.event)) quit = window.handleEvents();

		if (window.usesGPU() && computing) {
			rayTraceKernel<<<WIDTH,HEIGHT, 16384>>>(rayTracerGPU, d_data, d_world_ptr, world.objectsCount, d_cam, WIDTH, HEIGHT);
			gpuErrchk(cudaPeekAtLastError());
			cudaDeviceSynchronize();
			cudaMemcpy(data, d_data, TEXTURE_SIZE * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			//computing = false;
		}
		else if (computing){ 
			rayTracerCPU.rayTrace(data, &world, cam, WIDTH, HEIGHT); //rayCastingCPU(data, WIDTH, HEIGHT);
		}

		window.update(data);
		window.fpsLimiter.endFrame();
		window.fpsLimiter.delay();
		window.updateWindowTitle();
	}
	delete redMat;
	delete greenMat;
	delete blueMat;
	delete grayMat;
	delete cam;
	delete[] data;

	cudaFree(d_rtGpu);
	cudaFree(d_cam);
	cudaFree(d_data);
	World::GpuFree(d_world_ptr);

	return 0;
}