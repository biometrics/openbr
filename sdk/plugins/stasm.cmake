set(BR_WITH_STASM OFF CACHE BOOL "Build with Stasm")

if(${BR_WITH_STASM})
  find_package(Stasm REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/stasm.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${Stasm_LIBS})

  if(WIN32)
    install(DIRECTORY ${Stasm_DIR}/lib/ DESTINATION bin)
  else()
    install(DIRECTORY ${Stasm_DIR}/lib/ DESTINATION lib)
  endif()

  install(DIRECTORY ${Stasm_DIR}/models/ DESTINATION share/openbr/models/stasm)
endif()
