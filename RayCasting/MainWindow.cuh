#ifndef _MAINWINDOW_
#define _MAINWINDOW_

#include <cstdlib>
#include <iostream>
#include <SDL.h>
#include "FpsLimiter.h"
#include "base.cuh"

class MainWindow
{
private:
	std::string title;
	bool usesgpu;
	const char* generateTitle();
public:
	SDL_Window* window;
	FpsLimiter fpsLimiter;
	SDL_Renderer* renderer;
	SDL_Rect mainRect;
	SDL_Texture* mainTexture;
	SDL_Event event;

	MainWindow();
	~MainWindow();

	void update(Uint32* data);
	void updateWindowTitle();
	bool handleEvents();
	bool usesGPU();
};

#endif

