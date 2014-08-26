set(BR_WITH_STASM4 ON CACHE BOOL "Build with Stasm")

if(${BR_WITH_STASM4})
  find_package(Stasm4 REQUIRED)

  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/stasm4.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${Stasm4_LIBS})

  if(WIN32)
    install(FILES ${Stasm4_LIBS} DESTINATION bin)
  else()
    install(FILES ${Stasm4_LIBS} DESTINATION lib)
  endif()

  install(DIRECTORY ${Stasm4_DIR}/data/ DESTINATION share/openbr/models/stasm)
endif()
