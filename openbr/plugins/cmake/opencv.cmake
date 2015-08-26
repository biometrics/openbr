option(BR_WITH_OPENCV_CONTRIB "Build with OpenCV contrib plugins." ON)
if(${BR_WITH_OPENCV_CONTRIB})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} opencv_contrib)
  set(OPENCV_DEPENDENCIES ${OPENCV_DEPENDENCIES} opencv_contrib)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/imgproc/heatmap.cpp
                                                 plugins/imgproc/shapeaxisratio.cpp)
endif()

option(BR_WITH_OPENCV_FEATURES2D "Build with OpenCV features2d plugins." ON)
if(${BR_WITH_OPENCV_FEATURES2D})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} opencv_features2d)
  set(OPENCV_DEPENDENCIES ${OPENCV_DEPENDENCIES} opencv_features2d)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/distance/keypointmatcher.cpp
                                                 plugins/imgproc/keypointdescriptor.cpp
                                                 plugins/metadata/keypointdetector.cpp)
endif()

option(BR_WITH_OPENCV_FLANN "Build with OpenCV flann plugins." ON)
if(${BR_WITH_OPENCV_FLANN})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} opencv_flann)
  set(OPENCV_DEPENDENCIES ${OPENCV_DEPENDENCIES} opencv_flann)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/cluster/kmeans.cpp
                                                 plugins/cluster/randomcentroids.cpp)
endif()

option(BR_WITH_OPENCV_NONFREE "Build with OpenCV nonfree plugins." ON)
if(${BR_WITH_OPENCV_NONFREE})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} opencv_nonfree)
  set(OPENCV_DEPENDENCIES ${OPENCV_DEPENDENCIES} opencv_nonfree)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/imgproc/custom_sift.cpp
                                                 plugins/imgproc/sift.cpp)
endif()

option(BR_WITH_OPENCV_PHOTO "Build with OpenCV photo plugins." ON)
if(${BR_WITH_OPENCV_PHOTO})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} opencv_photo)
  set(OPENCV_DEPENDENCIES ${OPENCV_DEPENDENCIES} opencv_photo)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/imgproc/denoising.cpp
                                                 plugins/imgproc/inpaint.cpp)
endif()

option(BR_WITH_OPENCV_VIDEO "Build with OpenCV video plugins." ON)
if(${BR_WITH_OPENCV_VIDEO})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} opencv_video)
  set(OPENCV_DEPENDENCIES ${OPENCV_DEPENDENCIES} opencv_video)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/imgproc/opticalflow.cpp)
endif()
