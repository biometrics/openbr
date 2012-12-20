include_directories(.)

# Build the command line interface
add_subdirectory(br)

# Build the gui sdk
add_subdirectory(openbr-gui)

# Build the gui interface
add_subdirectory(br-gui)

# Build examples/tests
add_subdirectory(examples)
