include_directories(.)

# Build the command line interface
add_subdirectory(br)

# Build the gui sdk
add_subdirectory(openbr-gui)

# Build the gui algorithm creator interface
add_subdirectory(br-creator)

# Build examples/tests
add_subdirectory(examples)
