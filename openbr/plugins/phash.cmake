set(BR_WITH_PHASH OFF CACHE BOOL "Build with pHash")

if(${BR_WITH_PHASH})
  find_package(pHash REQUIRED)
  find_package(CImg REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/phash.cpp ${PHASH_SRC})
endif()
