#include <sstream>
#include "MainWindow.cuh"
#include "Engine/RayTracerGPU.cuh"

MainWindow::MainWindow() : usesgpu(true), fpsLimiter(MAX_FPS)
{
	bool software = false;
	if (SDL_Init(SDL_INIT_VIDEO) == -1)
	{
		std::cout << SDL_GetError() << std::endl;
		exit(EXIT_FAILURE);
	}
	window = SDL_CreateWindow(generateTitle(), WINDOW_X, WINDOW_Y, WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
	if (window == NULL)
	{
		std::cout << SDL_GetError() << std::endl;
		exit(EXIT_FAILURE);
	}
	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	if (renderer == NULL)
	{
		std::cout << SDL_GetError() << std::endl;
		renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
		software = true;
	}
	if (renderer == NULL)
	{
		std::cout << SDL_GetError() << std::endl;
		exit(EXIT_FAILURE);
	}
	else if (software) std::cout << "WARNING: using software renderer" << std::endl;
	mainTexture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
	if (mainTexture == NULL)
	{
		std::cout << SDL_GetError() << std::endl;
		exit(EXIT_FAILURE);
	}
	mainRect.x = mainRect.y = 0;
	mainRect.w = WIDTH;
	mainRect.h = HEIGHT;
}


MainWindow::~MainWindow()
{
	SDL_DestroyTexture(mainTexture);
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
}

void MainWindow::update(Uint32* data)
{
	SDL_UpdateTexture(mainTexture, &mainRect, data, PIXEL_SIZE * WIDTH);
	SDL_RenderCopy(renderer, mainTexture, NULL, NULL);
	SDL_RenderPresent(renderer);
}

void MainWindow::updateWindowTitle()
{
	SDL_SetWindowTitle(window, generateTitle());
}

bool MainWindow::handleEvents()
{
	SDL_KeyboardEvent board;
	SDL_Keycode key;
	if (event.type == SDL_QUIT)
		return true;
	if (event.type == SDL_KEYDOWN)
	{
		board = event.key;
		key = board.keysym.sym;
		if (key == SDLK_SPACE)
		{
			usesgpu = !usesgpu;
		}
	}
	return false;
}

bool MainWindow::usesGPU()
{
	return usesgpu;
}

bool replace(std::string& str, const std::string& from, const std::string& to)
{
	size_t start_pos = str.find(from);
	if (start_pos == std::string::npos)
		return false;
	str.replace(start_pos, from.length(), to);
	return true;
}

const char* MainWindow::generateTitle()
{
	title = WINDOW_TITLE_FORMAT;
	std::string fps_count_key = FPS_COUNT_KEY;
	std::string engine_key = ENGINE_KEY;
	std::string engine = (usesGPU()) ? ENGINE_GPU : ENGINE_CPU;
	replace(title, fps_count_key, fpsLimiter.getFps());
	replace(title, engine_key, engine);
	return title.c_str();
}