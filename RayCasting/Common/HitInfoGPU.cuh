//#ifndef HIT_INFO_GPU
//#define HIT_INFO_GPU
//
//namespace geometries
//{
//	class GeometricObject;
//	class World;
//}
//
//#include "../Geometries/SphereGPU.cuh"
//#include "../Primitives/primitives.cuh"
//
//using geometries::SphereGPU;
//using geometries::World;
//using primitives::Vector3;
//using primitives::Ray;
//
//struct HitInfoGPU
//{
//	SphereGPU* hitObject;
//	World* world;
//	Vector3 normal;
//	Vector3 hitPoint;
//	Ray ray;
//
//	__host__ __device__ HitInfoGPU();
//};
//
//#endif