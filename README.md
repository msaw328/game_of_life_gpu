# Game of Life
Game of Life is a [cellular automaton](https://en.wikipedia.org/wiki/Cellular_automaton) with rules defined as follows:

1. If the cell is dead and has 3 alive neighbours, it becomes alive
2. If the cell is alive and has less than 2 alive neighbours, it dies
3. If the cell is alive and has 2 or 3 alive neighbours, it lives onto the next generation
4. If the cell is alive and has more than 3 alive neighbours (max 8), it dies

While there exist some variations, i've decided to use the original rules in my code.

# CUDA
CUDA is Nvidia's proprietary toolkit + set of libraries + language spec that provides control over general purpose computations on GPU 
([GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units)).

# How it works
In CUDA vocabulary, Host means CPU and Device means GPU.

C or C++ code, compiled by a regular C or C++ compiler creates a binary which is ran on host.
Host code then uses API exposed by the CUDA library to manage contexts, memory and send instructions.

CUDA, aside from being a toolkit and a set of libraries, is also a language, based heavily on C++. It can be compiled into a binary that can be ran by GPU with nvcc, a compiler provided as 
part of the CUDA toolkit.

When programming in CUDA, one can use Driver API, or Runtime API. The former requires a lot of boilerplate code and is mostly low level, but allows for a better control over and better 
understanding on how host and device communicate. In addition to that, it also allows for easier division of host code and device code, which is what i was aiming for. As a result, i've 
decided to use the Driver API. That being said, Runtime API is much simpler and requires less code, so its definitely a good fit for someone who wants to start programming with CUDA, but 
doesn't want to bother with setting up context, loading a GPU binary, etc.

Probably the best resource on CUDA is [CUDA toolkit documentation](http://docs.nvidia.com/cuda/), a part of Nvidia Developer Zone.

The open alternative to CUDA, called OpenCL is a standard developed by the Khronos group (which also looks after OpenGL).
I've decided to use CUDA over OpenCL because i have an Nvidia GPU, so it is possible that CUDA code will run faster than OpenCL, thanks to possible optimisations they may have taken.
That being said, both CUDA and OpenCL are viable solutions in the world of GPU computing.
