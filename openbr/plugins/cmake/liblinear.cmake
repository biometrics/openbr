set(BR_WITH_LIBLINEAR OFF CACHE BOOL "Build with LibLinear")

if(${BR_WITH_LIBLINEAR})
  find_package(LibLinear REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${LibLinear_SRC})
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/classification/liblinear.cpp)
endif()
