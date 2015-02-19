set(BR_WITH_STASM4 ON CACHE BOOL "Build with Stasm")

if(${BR_WITH_STASM4})
  find_package(Stasm4 REQUIRED)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${Stasm4_LIBS})

  if(WIN32)
    install(DIRECTORY ${Stasm_DIR}/build/ DESTINATION bin)
  else()
    install(DIRECTORY ${Stasm_DIR}/build/ DESTINATION lib)
  endif()

  install(DIRECTORY ${Stasm_DIR}/data/ DESTINATION share/openbr/models/stasm)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/metadata/stasm4.cpp)
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/imgproc/revertaffine.cpp)
endif()
