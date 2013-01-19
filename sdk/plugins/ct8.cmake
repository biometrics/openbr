set(BR_WITH_CT8 OFF CACHE BOOL "Build with Cognitec FaceVACS 8")

if(${BR_WITH_CT8})
  find_package(CT8 REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/ct8.cpp ${CT8_RESOURCES})

  set(CT8_LIBS optimized ${CT8_LIBRARY_RELEASE} debug ${CT8_LIBRARY_DEBUG})
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${CT8_LIBS})

  install(FILES ${CT8_DIR_LIB}/libfrsdk-8.5.0.so
                ${CT8_DIR_LIB}/../share/libhasp_linux_x86_64_67109.so
                ${CT8_DIR_LIB}/../share/libiomp5.so
          DESTINATION lib)
  install(DIRECTORY ${CT8_DIR}/etc DESTINATION models/ct8)
endif()
