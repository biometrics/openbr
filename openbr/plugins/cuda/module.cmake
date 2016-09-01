# add WITH_CUDA option
option(BR_WITH_CUDA "Build CUDA-accelerated plugins." OFF)
set(BR_CUDA_ARCH "sm_20" CACHE STRING "CUDA Architecture")

# only build this module if explicitly OK'ed
if(BR_WITH_CUDA)
  message(STATUS "Building with CUDA Support")
  find_package(CUDA REQUIRED)

  set(CUDA_SRC_DIR ${PROJECT_SOURCE_DIR}/openbr/plugins/cuda)

  # configure the compiler, need -fPIC for shared library
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=${BR_CUDA_ARCH} --compiler-options -fPIC)
  include_directories(${CUDA_INCLUDE_DIRS})

  # glob sources
  file(GLOB CUDA_CU_SRC ${CUDA_SRC_DIR}/*.cu)
  file(GLOB CUDA_CPP_SRC ${CUDA_SRC_DIR}/*.cpp)

  # compile each of the object files and append to CUDA_CU_OBJ
  foreach(FILE ${CUDA_CU_SRC})
    cuda_compile(FILE_O ${FILE})
    set(CUDA_CU_OBJ ${CUDA_CU_OBJ} ${FILE_O})
  endforeach()

  # ensure add_library knows these are external object file
  set_source_files_properties(${CUDA_CU_OBJ} PROPERTIES EXTERNAL_OBJECT true)

  # add the compiled source and libs into the build system
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${CUDA_CPP_SRC} ${CUDA_CU_OBJ})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${CUDA_LIBRARIES} "cublas" "cusolver")

endif()
