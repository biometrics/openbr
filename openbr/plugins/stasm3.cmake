set(BR_WITH_STASM3 OFF CACHE BOOL "Build with Stasm")

if(${BR_WITH_STASM3})
  find_package(Stasm3 REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/stasm3.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${Stasm3_LIBS})

  if(WIN32)
    install(DIRECTORY ${Stasm_DIR}/build/ DESTINATION bin)
  else()
    install(DIRECTORY ${Stasm_DIR}/build/ DESTINATION lib)
  endif()

  install(DIRECTORY ${Stasm_DIR}/models/ DESTINATION share/openbr/models/stasm)
endif()
