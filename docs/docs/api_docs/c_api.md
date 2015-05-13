# C API

The C API is a high-level API for running algorithms and evaluating results.

In order to provide a high-level interface that is usable from the command line and callable from other programming languages, the API is designed to operate at the "file system" level.
In other words, arguments to many functions are file paths that specify either a source of input or a desired output.
File extensions are relied upon to determine *how* files should be interpreted in the context of the function being called.
The [C++ Plugin API](cpp_api.md) should be used if more fine-grained control is required.

## Important API Considerations

Name | Consideration
--- | ---
<a class="table-anchor" id="memory"></a>Memory | Memory for <tt>const char*</tt> return values is managed internally and guaranteed until the next call to the function
<a class="table-anchor" id="input-string-buffers"></a>Input String Buffers | Users should input a char * buffer and the size of that buffer. String data will be copied into the buffer, if the buffer is too small, only part of the string will be copied. Returns the buffer size required to contain the complete string.


## Using the API

To use the API in your project include the following file:

    #include <openbr/openbr.h>

[CMake](http://www.cmake.org/) developers may wish to the cmake configuration file found at:

    share/openbr/cmake/OpenBRConfig.cmake

Please see the [tutorials](../tutorials.md) section for examples.
