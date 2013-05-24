set(BR_WITH_EIGEN3 ON CACHE BOOL "Build Eigen3 plugins")

if(${BR_WITH_EIGEN3})
  find_package(Eigen3 REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/eigen3.cpp)
  install(FILES ${EIGEN3_LICENSE} RENAME Eigen3 DESTINATION share/openbr/licenses)
endif()
