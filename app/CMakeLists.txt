# Build the command line interface
add_subdirectory(br)

if(NOT BR_PACKAGE_THIRDPARTY)
  if(NOT BR_EMBEDDED)
    # Build examples/tests
    add_subdirectory(examples)

    # Build additional OpenBR utilities
    add_subdirectory(br-gui)
  endif()
endif()
