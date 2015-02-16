set(BR_WITH_DLIB OFF CACHE BOOL "Build with LibLinear")

message(${BR_WITH_DLIB})

if(${BR_WITH_DLIB})
  find_package(DLib REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/dlib.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${DLib_LIBS})

  install(DIRECTORY ${DLib_DIR}/models/ DESTINATION share/openbr/models/dlib)
endif()
