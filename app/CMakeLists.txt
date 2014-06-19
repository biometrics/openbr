# Build the command line interface
add_subdirectory(br)

# Build examples/tests
add_subdirectory(examples)

# Build additional OpenBR utilities
if(NOT ${BR_EMBEDDED})
  add_subdirectory(br-crawl)
  add_subdirectory(br-gui)
  add_subdirectory(br-search)
  add_subdirectory(br-serve)
endif()
