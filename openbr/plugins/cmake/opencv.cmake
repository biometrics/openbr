option(BR_WITH_OPENCV_FLANN "Build with OpenCV flann plugins." ON)
if(${BR_WITH_OPENCV_FLANN})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} opencv_flann)
  set(OPENCV_DEPENDENCIES ${OPENCV_DEPENDENCIES} opencv_flann)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/cluster/kmeans.cpp
                                                 plugins/cluster/randomcentroids.cpp)
endif()

option(BR_WITH_OPENCV_OBJDETECT "Build with OpenCV objdetect plugins." ON)
if(${BR_WITH_OPENCV_OBJDETECT})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} opencv_objdetect)
  set(OPENCV_DEPENDENCIES ${OPENCV_DEPENDENCIES} opencv_objdetect)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/imgproc/hog.cpp
                                                 plugins/metadata/cascade.cpp
                                                 plugins/metadata/hogpersondetector.cpp)
endif()
