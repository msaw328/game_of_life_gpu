#ifndef GRID_H_
#define GRID_H_

#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "range.h"

#define CELL_ALIVE 1
#define CELL_DEAD 0

// device code for getting and setting field buffer

typedef struct {
    range_t x_axis;
    range_t y_axis;
    int exc;
    int* buffer;

    // cuda state
    CUdevice dev;
    CUcontext ctx;
    CUmodule cuda_mod;
    CUdeviceptr dev_buffer;
} grid_t;

CUresult grid_make(grid_t* g, range_t x, range_t y, int excessive_cells, const char* cuda_mod_name, CUdevice device);

CUresult grid_destroy(grid_t* g);

// alive, dead, or -1 if out of bounds
int grid_get(grid_t* g, int x, int y);
int grid_set(grid_t* g, int x, int y, int val);

CUresult grid_gpu_init_state(grid_t* g);

CUresult grid_gpu_request_next_generation(grid_t* g);

#endif
