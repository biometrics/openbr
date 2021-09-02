set(BR_WITH_DLIB OFF CACHE BOOL "Build with DLib")

if(${BR_WITH_DLIB})
  ExternalProject_Add(dlib
                      URL http://downloads.sourceforge.net/project/dclib/dlib/v18.16/dlib-18.16.tar.bz2
                      URL_MD5 e9e5449bc25370afce2d254327afac99
                      SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/dlib-18.16"
                      CONFIGURE_COMMAND ""
                      BUILD_COMMAND ""
                      INSTALL_COMMAND "")
  include_directories("${PROJECT_SOURCE_DIR}/3rdparty/dlib-18.16")
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/classification/dlib.cpp plugins/classification/dlib_source.cpp)
endif()
