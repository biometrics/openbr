# Build the command line interface
add_subdirectory(br)

if (NOT BR_PACKAGE_THIRDPARTY)
# Build examples/tests
add_subdirectory(examples)

# Build additional OpenBR utilities
if(NOT ${BR_EMBEDDED})
  add_subdirectory(br-gui)
endif()
endif()
