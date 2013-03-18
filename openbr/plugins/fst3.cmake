set(BR_WITH_FST3 OFF CACHE BOOL "Build with Feature Selection Toolbox 3")

if(${BR_WITH_FST3})
  find_package(FST3 REQUIRED)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} plugins/fst3.cpp ${FST3_SRC})

  find_package(Boost REQUIRED)
  include_directories(${Boost_INCLUDE_DIRS})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} boost_thread)

  find_package(LibSVM REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${LibSVM_SRC})
endif()
