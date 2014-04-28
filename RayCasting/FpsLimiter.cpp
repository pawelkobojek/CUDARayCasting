#include <SDL.h>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include "FpsLimiter.h"

FpsLimiter::FpsLimiter(unsigned maxFps)
{
	this->maxFps = maxFps;
	frames = 0;
	lastFps = 0;
	startFpsTimer = clock();
}

FpsLimiter::~FpsLimiter()
{
}

void FpsLimiter::startFrame()
{
	frames++;
	drawStartTime = clock();
}

void FpsLimiter::endFrame()
{
	int drawEndTime = clock();
	delayToNextFrame = (CLOCKS_PER_SEC / maxFps) - (drawEndTime - drawStartTime);
	delayToNextFrame = (clock_t)floor(delayToNextFrame + 0.5);
	delayToNextFrame < 0 ? delayToNextFrame = 0 : NULL;
	if (clock() - startFpsTimer>1000)
	{
		lastFps = frames;
		frames = 0;
		startFpsTimer = clock();
	}
}

void FpsLimiter::delay()
{
	SDL_Delay(delayToNextFrame);
}

int FpsLimiter::getRawFps()
{
	return lastFps;
}

std::string FpsLimiter::getFps()
{
	std::stringstream ss;
	ss << lastFps;
	return ss.str();
}