#include "PerfectDiffuse.cuh"

namespace materials
{
	PerfectDiffuse::PerfectDiffuse(ColorRGB c) : materialColor(c) {}

		ColorRGB PerfectDiffuse::radiance(const PointLight light, const HitInfo hit) const
	{
			Vector3 inD = (light.getPosition() - hit.hitPoint).normalized();
			float diffFactor = inD.dot(hit.normal);
			if (diffFactor < 0) return ColorRGB(0.0f, 0.0f, 0.0f);//;return ColorRGB::black;

				return light.getColor() * materialColor * diffFactor;
		}
}