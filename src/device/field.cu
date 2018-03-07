#include "field.h"

extern "C" {
    __device__ int get_field_index(int x, int y, int width) {
        return x + y * width;
    }

    __global__ void next_generation(int* old_gen, int* new_gen, int width, int height) {
        int alive_counter = 0;

        for(int i = -1; i < 2; i++) {
            for(int j = -1; j < 2; j++) {
                int tmp_x = threadIdx.x + i;
                int tmp_y = threadIdx.y + j;

                // edge cases
                if(i == 0 && j == 0) continue;
                if(tmp_x < 0 || width - 1 < tmp_x) continue;
                if(tmp_y < 0 || height - 1 < tmp_y) continue;

                // else
                alive_counter += old_gen[get_field_index(tmp_x, tmp_y, width)];
            }
        }

        int current_index = get_field_index(threadIdx.x, threadIdx.y, width);
        if(old_gen[current_index] == CELL_DEAD) {
            if(alive_counter == 3) {
                new_gen[current_index] = CELL_ALIVE;
            } else {
                new_gen[current_index] = CELL_DEAD;
            }
        } else {
            if(alive_counter == 2 || alive_counter == 3) {
                new_gen[current_index] = CELL_ALIVE;
            } else {
                new_gen[current_index] = CELL_DEAD;
            }
        }
    }
}
