if(NOT ${BR_EMBEDDED})
  find_package(Qwt REQUIRED)
endif()

include_directories(.)

# Build the command line interface
add_subdirectory(br)

# Build the gui sdk
add_subdirectory(openbr-gui)

# Build forensic face
add_subdirectory(forensicface)

# Build the gui algorithm creator interface
add_subdirectory(br-creator)

# Build examples/tests
add_subdirectory(examples)
