set(BR_WITH_LIKELY OFF CACHE BOOL "Build with Likely")

if(${BR_WITH_LIKELY})
  find_package(Likely REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/likely.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${Likely_LIBS})
endif()
