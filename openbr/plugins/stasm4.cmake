set(BR_WITH_STASM4 OFF CACHE BOOL "Build with Stasm")

if(${BR_WITH_STASM4})
  find_package(Stasm4 REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/stasm4.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${Stasm4_LIBS})

  if(WIN32)
    install(DIRECTORY ${Stasm_DIR}/build/ DESTINATION bin)
  else()
    install(DIRECTORY ${Stasm_DIR}/build/ DESTINATION lib)
  endif()

  #install(DIRECTORY ${Stasm_DIR}/models/ DESTINATION share/openbr/models/stasm)
endif()
