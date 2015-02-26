set(BR_WITH_LIBAV OFF CACHE BOOL "Build with LibAV")

if(${BR_WITH_LIBAV})
  find_package(LibAV REQUIRED)
  include_directories(${LIBAV_INCLUDE_DIR})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${LIBAV_LIBRARIES})
  
  foreach(LIBAV_LIB ${LIBAV_LIBRARIES})
    install(FILES ${LIBAV_LIB} DESTINATION lib)
  endforeach()
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/gallery/keyframes.cpp)
endif()
