set(BR_WITH_STASM4 ON CACHE BOOL "Build with Stasm")

if(${BR_WITH_STASM4})
  find_package(Stasm4 REQUIRED)

  include_directories(${STASM4_DIR}/include)

  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/stasm4.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${STASM4_LIBS})

  if(WIN32)
    install(DIRECTORY ${STASM4_DIR}/bin/ DESTINATION bin)
    install(DIRECTORY ${STASM4_DIR}/lib/ DESTINATION lib)
  else()
    install(DIRECTORY ${STASM4_DIR}/lib/ DESTINATION lib)
  endif()

  install(DIRECTORY ${STASM4_INCLUDE_DIR}/ DESTINATION include)
  install(DIRECTORY ${STASM4_DATA_DIR} DESTINATION share/openbr/models/stasm)
endif()
