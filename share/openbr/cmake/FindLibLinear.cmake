find_path(LibLinear_DIR linear.h ${CMAKE_SOURCE_DIR}/3rdparty/*)

mark_as_advanced(LibLinear_DIR)
include_directories(${LibLinear_DIR})
include_directories(${LibLinear_DIR}/blas)

set(LibLinear_SRC ${LibLinear_DIR}/linear.cpp 
					${LibLinear_DIR}/tron.cpp
					${LibLinear_DIR}/blas/daxpy.c
					${LibLinear_DIR}/blas/ddot.c
					${LibLinear_DIR}/blas/dnrm2.c
					${LibLinear_DIR}/blas/dscal.c)
