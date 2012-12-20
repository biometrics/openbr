# Build SWIG Python wrapper?
set(BR_SDK_SWIG OFF CACHE BOOL "Enable SWIG wrappers.")

if(${BR_SDK_SWIG})
  # Find SWIG
  find_package(SWIG REQUIRED)
  include(${SWIG_USE_FILE})

  # Find Python
  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_PATH})

  # Find Java
  find_package(JNI REQUIRED)
  include_directories(${JNI_INCLUDE_DIRS})

  # Configure SWIG
  set(CMAKE_SWIG_FLAGS "")
  set_source_files_properties(br_sdk_swig.i PROPERTIES CPLUSPLUS ON SWIG_FLAGS "-includeall")

  # Build SWIG Python
  swig_add_module(br_sdk_swig python br_sdk_swig.i)
  swig_link_libraries(br_sdk_swig br_sdk ${PYTHON_LIBRARIES})

  install(CODE "file( GLOB _GeneratedPythonSources \"${CMAKE_CURRENT_BINARY_DIR}/*.py\" )"
          CODE "file( INSTALL \${_GeneratedPythonSources} DESTINATION \"include/br/python\" )")
  install(TARGETS _br_sdk_swig
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)

  # Build SWIG Java
  swig_add_module(br_sdk_swig java br_sdk_swig.i)
  swig_link_libraries(br_sdk_swig br_sdk ${JNI_LIBRARIES})

  install(CODE "file( GLOB _GeneratedJavaSources \"${CMAKE_CURRENT_BINARY_DIR}/*.java\" )"
          CODE "file( INSTALL \${_GeneratedJavaSources} DESTINATION \"include/br/java\" )")
  install(TARGETS br_sdk_swig
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
endif()
