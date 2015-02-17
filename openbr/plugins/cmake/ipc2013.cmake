set(BR_WITH_IPC2013 OFF CACHE BOOL "Build with Intel Perceptual Computing SDK 2013")

if(${BR_WITH_IPC2013})
  find_package(IPC2013 REQUIRED)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${IPC2013_LIBS})
  install(DIRECTORY ${IPC2013_DIR}/bin/x64/ DESTINATION bin)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/classification/ipc2013.cpp)
endif()
