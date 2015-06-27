set(BR_WITH_CAFFE OFF CACHE BOOL "Build with Caffe")

if (${BR_WITH_CAFFE})
   find_package(Caffe)
   set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${Caffe_LIBRARIES})
   include_directories(${Caffe_INCLUDE_DIRS})
else()
   set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/gallery/lmdbGallery.cpp)
endif()

