set(BR_WITH_PP5 OFF CACHE BOOL "Build with PittPatt 5")

if(${BR_WITH_PP5})
  find_package(PP5 REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/pp5.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${PP5_LIBS})

  if(WIN32)
    install(DIRECTORY ${PP5_DIR}/lib/ DESTINATION bin)
  else()
    install(DIRECTORY ${PP5_DIR}/lib/ DESTINATION lib)
  endif()

  install(DIRECTORY ${PP5_DIR}/models/ DESTINATION share/openbr/models/pp5)
endif()
