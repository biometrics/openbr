set(BR_WITH_PP4 ON CACHE BOOL "Build with PittPatt 4")

if(${BR_WITH_PP4})
  find_package(PP4 REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/pp4.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${PP4_LIBS})
  install(DIRECTORY ${PP4_DIR}/lib/ DESTINATION lib)
  install(DIRECTORY ${PP4_DIR}/models/ DESTINATION models/pp4)
endif()
