set(BR_WITH_NT4 OFF CACHE BOOL "Build with Neurotec Biometric 4")

if(${BR_WITH_NT4})
  find_package(NT4 REQUIRED)
  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} plugins/nt4.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${NT4_LIBS})
  install(DIRECTORY ${NT4_DIR_LIB}/ DESTINATION lib)
endif()
