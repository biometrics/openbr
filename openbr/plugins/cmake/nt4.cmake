set(BR_WITH_NT4 OFF CACHE BOOL "Build with Neurotec Biometric 4")

if(${BR_WITH_NT4})
  find_package(NT4 REQUIRED)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${NT4_LIBS})
  install(DIRECTORY ${NT4_DIR_LIB}/ DESTINATION lib)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/classification/nt4.cpp)
endif()
