#include "range.h"

void range_make(range_t* r, int min, int max) {
    r->min = min;
    r->max = max;
}

int range_width(range_t r) {
    return r.max - r.min + 1;
}

int range_contains(range_t r, int n) {
    return (n >= r.min && n <= r.max);
}

int range_get(range_t r, int n) {
    return r.min + n;
}

int range_index(range_t r, int n) {
    return n - r.min;
}
