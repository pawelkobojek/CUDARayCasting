#ifndef _BASE_
#define WINDOWS_LEAN_AND_MEAN

#include <SDL.h>

#define _BASE_

#define WINDOW_X 100
#define WINDOW_Y 100
#define WIDTH 400
#define HEIGHT 400
#define DEPTH 32
#define PIXEL_SIZE 4
#define TEXTURE_SIZE WIDTH*HEIGHT
#define TEXTURE_BYTES_COUNT TEXTURE_SIZE*PIXEL_SIZE
#define COORDS_2DTO1D(X,Y) (Y*WIDTH+X)
#define DH_CALLABLE __host__ __device__
#define MAX_FPS 600

#define COLOR(r,g,b,a) (r<<24 | g<<16 | b<<8 | a)
#define R(color) (color>>24)
#define G(color) ((color>>16)&0x000000ff)
#define B(color) ((color>>8)&0x000000ff)
#define A(color) (color&0x000000ff)

#define WINDOW_TITLE_FORMAT "Ray casting ({ENGINE}) FPS:{FPS_COUNT}"
#define FPS_COUNT_KEY "{FPS_COUNT}"
#define ENGINE_KEY "{ENGINE}"
#define ENGINE_CPU "CPU"
#define ENGINE_GPU "GPU"
#endif