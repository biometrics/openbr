# Build the command line interface
if(NOT BR_PACKAGE_THIRDPARTY)
  if(NOT BR_EMBEDDED)
    add_subdirectory(br)

    # Build examples/tests
    add_subdirectory(examples)

    # Build additional OpenBR utilities
    add_subdirectory(br-gui)
  endif()
endif()
