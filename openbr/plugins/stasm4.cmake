set(BR_WITH_STASM4 ON CACHE BOOL "Build with Stasm")

if(${BR_WITH_STASM4})
  find_package(Stasm4 REQUIRED)

  include_directories(${STASM4_INCLUDE_DIR})

  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/stasm4.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${STASM4_LIBS})

  if(WIN32)
    install(FILES ${STASM4_LIBS} DESTINATION bin)
  else()
    install(FILES ${STASM4_LIBS} DESTINATION lib)
  endif()

  install(DIRECTORY ${STASM4_DATA_DIR} DESTINATION share/openbr/models/stasm)
endif()
