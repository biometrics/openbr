set(BR_WITH_PBD OFF CACHE BOOL "Build with PartsBasedDetector")

if(${BR_WITH_PBD})
  find_package(PBD REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/pbd.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${PBD_LIBS})

  if(WIN32)
    install(DIRECTORY ${PBD_DIR}/lib/ DESTINATION bin)
  else()
    install(DIRECTORY ${PBD_DIR}/lib/ DESTINATION lib)
  endif()

  install(DIRECTORY ${PBD_DIR}/models/ DESTINATION share/openbr/models/pbd)
endif()
