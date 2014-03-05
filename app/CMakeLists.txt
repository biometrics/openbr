# Build the command line interface
add_subdirectory(br)

# Build examples/tests
add_subdirectory(examples)

# Build OpenBR GUI application
if(NOT ${BR_EMBEDDED})
  add_subdirectory(br-gui)
endif()
