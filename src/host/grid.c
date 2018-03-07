#include "grid.h"

#include <stdio.h>

size_t __grid_size(grid_t* g) {
    return (g->exc * 2 + range_width(g->x_axis)) * (g->exc * 2 + range_width(g->y_axis)) * sizeof(int);
}

CUresult grid_make(grid_t* g, range_t x, range_t y, int excessive_cells, const char* cuda_mod_name, CUdevice dev) {
    g->x_axis = x;
    g->y_axis = y;
    g->exc = excessive_cells;

    size_t buf_size = __grid_size(g);

    g->buffer = malloc(buf_size);
    memset(g->buffer, CELL_DEAD, buf_size);

    // cuda
    CUresult res;
    g->dev = dev;

    if((res = cuCtxCreate(&(g->ctx), 0, g->dev)) != CUDA_SUCCESS) {
        return res;
    }

    if((res = cuModuleLoad(&(g->cuda_mod), cuda_mod_name)) != CUDA_SUCCESS) {
        return res;
    }

    return CUDA_SUCCESS;
}

CUresult grid_destroy(grid_t* g) {
    free(g->buffer);

    CUresult res;
    if((res = cuModuleUnload(g->cuda_mod)) != CUDA_SUCCESS) {
        return res;
    }

    return cuCtxDetach(g->ctx);
}

int grid_get(grid_t* g, int x, int y) {
    if(range_contains(g->x_axis, x) && range_contains(g->y_axis, y)) {
        int real_x = range_index(g->x_axis, x); // translates range vlaues to array indices
        int real_y = range_index(g->y_axis, y);

        int real_width = range_width(g->x_axis) + 2 * g->exc;
        real_x = (real_x + g->exc);
        real_y = (real_y + g->exc);

        //printf("X: %d Y: %d ==> %d\n", real_x, real_y, real_x + real_width * real_y);

        return g->buffer[real_x + real_width * real_y];
    } else {
        return -1;
    }
}

int grid_set(grid_t* g, int x, int y, int val) {
    if(range_contains(g->x_axis, x) && range_contains(g->y_axis, y)) {
        int real_x = range_index(g->x_axis, x); // translates range values to array indices
        int real_y = range_index(g->y_axis, y);
        
        int real_width = range_width(g->x_axis) + 2 * g->exc;
        real_x = (real_x + g->exc);
        real_y = (real_y + g->exc);

        g->buffer[real_x + real_width * real_y] = val;
        return 0;
    } else {
        return -1;
    }
}

CUresult grid_gpu_init_state(grid_t* g) {
    CUresult res;
    if((res = cuMemAlloc(&(g->dev_buffer), __grid_size(g))) != CUDA_SUCCESS) {
        return res;
    }

    if((res = cuMemcpyHtoD(g->dev_buffer, g->buffer, __grid_size(g))) != CUDA_SUCCESS) {
        return res;
    }

    return CUDA_SUCCESS;
}

CUresult grid_gpu_request_next_generation(grid_t* g) {
    CUresult res;
    CUfunction next_generation;

    CUdeviceptr next_generation_buffer;

    if((res = cuMemAlloc(&next_generation_buffer, __grid_size(g))) != CUDA_SUCCESS) {
        return res;
    }

    if((res = cuModuleGetFunction(&next_generation, g->cuda_mod, "next_generation")) != CUDA_SUCCESS) {
        return res;
    }

    int width = range_width(g->x_axis) + 2 * g->exc;
    int height = range_width(g->y_axis) + 2 * g->exc;

    void* args[4] = { &(g->dev_buffer), &next_generation_buffer, &width, &height };

    res = cuLaunchKernel(next_generation,
                            1, 1, 1, // blocks
                            width, height, 1, // threads
                            0, 0, args, 0);

    if(res != CUDA_SUCCESS) {
        return res;
    }

    if((res = cuMemFree(g->dev_buffer)) != CUDA_SUCCESS) {
        return res;
    }

    g->dev_buffer = next_generation_buffer;

    if((res = cuMemcpyDtoH(g->buffer, g->dev_buffer, __grid_size(g))) != CUDA_SUCCESS) {
        return res;
    }
}
