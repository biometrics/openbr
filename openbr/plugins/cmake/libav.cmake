set(BR_WITH_LIBAV OFF CACHE BOOL "Build with LibAV")

if(${BR_WITH_LIBAV})
  if(WIN32 AND NOT LIBAV_ROOT_DIR)
    ExternalProject_Add(libav
                        URL http://builds.libav.org/windows/release-lgpl/libav-x86_64-w64-mingw32-11.2.7z
                        URL_MD5 902abcf8d1562adcc367424889a81b36
                        SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/libav"
                        CONFIGURE_COMMAND ""
                        BUILD_COMMAND ""
                        INSTALL_COMMAND "")
    set(LIBAV_ROOT_DIR "${PROJECT_SOURCE_DIR}/3rdparty/libav/usr")
    set(LIBAV_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/libav/usr/include")
    list(APPEND LIBAV_LIBRARIES "${PROJECT_SOURCE_DIR}/3rdparty/libav/usr/bin/avcodec.lib"
                                "${PROJECT_SOURCE_DIR}/3rdparty/libav/usr/bin/avdevice.lib"
                                "${PROJECT_SOURCE_DIR}/3rdparty/libav/usr/bin/avformat.lib"
                                "${PROJECT_SOURCE_DIR}/3rdparty/libav/usr/bin/avutil.lib"
                                "${PROJECT_SOURCE_DIR}/3rdparty/libav/usr/bin/swscale.lib")
  else()
    find_package(LibAV REQUIRED)
  endif()

  include_directories(${LIBAV_INCLUDE_DIR})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${LIBAV_LIBRARIES})
  
  foreach(LIBAV_LIB ${LIBAV_LIBRARIES})
    install(FILES ${LIBAV_LIB} DESTINATION lib)
  endforeach()
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/gallery/keyframes.cpp)
endif()
