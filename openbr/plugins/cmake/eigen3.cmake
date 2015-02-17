set(BR_WITH_EIGEN3 ON CACHE BOOL "Build Eigen3 plugins")

if(${BR_WITH_EIGEN3})
  find_package(Eigen3 REQUIRED)
  install(FILES ${EIGEN3_LICENSE} RENAME Eigen3 DESTINATION share/openbr/licenses)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/classification/lda.cpp)
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/imgproc/revertaffine.cpp)
endif()
