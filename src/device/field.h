#ifndef FIELD_H_
#define FIELD_H_

#define CELL_ALIVE 1
#define CELL_DEAD 0

// device code for working on field buffers

extern "C" {
    __device__ int get_field_index(int x, int y, int width);
    __global__ void next_generation(int* old_gen, int* new_gen, int width, int height);
}

#endif
