set(BR_WITH_DLIB OFF CACHE BOOL "Build with DLib")

if(${BR_WITH_DLIB})
  find_package(DLib REQUIRED)

  add_definitions(-DDLIB_NO_GUI_SUPPORT)

  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/classification/dlib.cpp)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${DLib_DIR}/dlib/all/source.cpp)

  install(DIRECTORY ${DLib_DIR}/models/ DESTINATION share/openbr/models/dlib)
endif()
