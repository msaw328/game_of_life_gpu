#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#include "grid.h"

// pretty print
void print_grid(grid_t* g);

int main(int argc, char** argv) {
    // cuda boilerplate
    CUresult res = cuInit(0);
    if(res != CUDA_SUCCESS) {
        printf("CUDA error after cuInit\n");
        exit(0);
    }

    int dev_count;
    res = cuDeviceGetCount(&dev_count);

    if(dev_count == 0) {
        printf("No CUDA compatible devices found\n");
        exit(0);
    }

    CUdevice dev;
    res = cuDeviceGet(&dev, 0);
    if(res != CUDA_SUCCESS) {
        printf("CUDA error after cuDeviceGet\n");
        exit(0);
    }

    // actual code
    range_t x;
    range_make(&x, -7, 7);

    range_t y;
    range_make(&y, -7, 7);

    int excessive_cells = 2;

    int generations = 150;

    grid_t g;
    grid_make(&g, x, y, excessive_cells, "kernels/field.cubin", dev);

    // glider 1
    grid_set(&g, 0, 0, CELL_ALIVE);
    grid_set(&g, 0, 1, CELL_ALIVE);
    grid_set(&g, 0, -1, CELL_ALIVE);
    grid_set(&g, -1, 0, CELL_ALIVE);
    grid_set(&g, 1, 1, CELL_ALIVE);


    printf("generation 0\n\n");
    print_grid(&g);
    printf("\n=====\n\n");

    grid_gpu_init_state(&g);

    for(int i = 1; i < generations; i++) {
        sleep(1);
        grid_gpu_request_next_generation(&g);
        printf("generation %d\n\n", i);
        print_grid(&g);
        printf("\n=====\n\n");
    }

    grid_destroy(&g);
}

void print_grid(grid_t* g) {
    for(int j = g->y_axis.max; j >= g->y_axis.min; j--) {
        for(int i = g->x_axis.min; i <= g->x_axis.max; i++) {
            printf("%s ", (grid_get(g, i, j)) ? "X" : ".");
        }
        printf("\n");
    }
}
