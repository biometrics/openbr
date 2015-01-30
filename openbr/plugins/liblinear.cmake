set(BR_WITH_LIBLINEAR OFF CACHE BOOL "Build with LibLinear")

if(${BR_WITH_LIBLINEAR})
  find_package(LibLinear REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${LibLinear_SRC})
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/liblinear.cpp)
endif()
