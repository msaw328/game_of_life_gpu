#ifndef RANGE_H_
#define RANGE_H_

typedef struct {
    int min;
    int max;
} range_t;

void range_make(range_t* r, int min, int max);

int range_width(range_t r);

int range_contains(range_t r, int n);

// those two functions have to be externally
// bound checked with range_width() or range_contains()
int range_get(range_t r, int n);
int range_index(range_t r, int n);

#endif
