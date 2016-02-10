# CUDA Plugins
This folder contains CUDA-accelerated OpenBR plugins.  They are structured in the following format.

## File Structure
We will use a plugin called `CUDAPlugin` as an example.

Each plugin has 3 files associated with it: a CUDA file, CPP file, and HPP header file.
```
cudaplugin.cu
cudaplugin.cpp
cudaplugin.hpp
```
The `.cu` file contains CUDA kernel functions and the corresponding wrapper functions
that directly call the kernel functions.  The `.cpp` files contain the OpenBR
standard plugin declaration.  Functions in this file call the wrappers.  The `.hpp`
contains header declarations for the CUDA wrapper functions so the `.cpp` file
knows how to call them.

# CUDA Files
All functions for a particular CUDA plugin are defined in a namespace of that
plugin's name which is defined within `br::cuda` namespace.  For example, if
we have a plugin called CUDAPlugin, both wrapper and kernel functions should
be globally defined within `br::cuda::cudaplugin`.
