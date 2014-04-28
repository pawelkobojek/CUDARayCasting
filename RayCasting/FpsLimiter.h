#ifndef _FPSLIMITER_
#define _FPSLIMITER_

#include <string>
#include <cstdlib>
#include <ctime>

class FpsLimiter
{
	int frames;
	int lastFps;
	clock_t drawStartTime;
	clock_t delayToNextFrame;
	clock_t startFpsTimer;
public:
	unsigned maxFps;
	void startFrame();
	void endFrame();
	void delay();
	std::string getFps();
	int getRawFps();
	
	FpsLimiter(unsigned maxFps);
	~FpsLimiter();
};

#endif