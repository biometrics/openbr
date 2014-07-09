# Build the examples...

# Loop over all the examples, specifying the source and library.
add_custom_target (examples ALL)
add_definitions (${PROJECT_DEFINITIONS})

foreach (EXAMPLE ${EXAMPLES})

  add_executable (${EXAMPLE} ${EXAMPLE}.cpp)
  add_dependencies (examples ${EXAMPLE})
  target_link_libraries (${EXAMPLE} ${PROJECT_LIBRARIES})

endforeach ()

if (MSVC OR CMAKE_CONFIGURATION_TYPES)
  # Add _d suffix for your debug versions of the tools
  set_target_properties (${EXAMPLES} PROPERTIES
    DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
endif ()

# RandomSave uses boost serialization if it's available
if (Boost_FOUND AND Boost_SERIALIZATION_FOUND)
  include_directories (${Boost_INCLUDE_DIRS})
  set_target_properties (RandomSave PROPERTIES
    COMPILE_DEFINITIONS HAVE_BOOST_SERIALIZATION=1)
  target_link_libraries (RandomSave ${Boost_LIBRARIES})
endif ()

# RandomThread uses OpenMP if it's available
if (OPENMP_FOUND)
  set_target_properties (RandomThread PROPERTIES
    COMPILE_FLAGS ${OpenMP_CXX_FLAGS} COMPILE_DEFINITIONS HAVE_OPENMP=1)
  if (NOT MSVC)
    set_target_properties (RandomThread
      PROPERTIES LINK_FLAGS ${OpenMP_CXX_FLAGS})
  endif ()
endif ()

if (HAVE_MPFR)
  add_executable (MPFRExample MPFRExample.cpp)
  add_dependencies (examples MPFRExample)
  target_link_libraries (MPFRExample ${MPFR_LIBRARIES} ${GMP_LIBRARIES})
endif ()

if (MSVC)
  get_target_property (_LIBTYPE ${PROJECT_LIBRARIES} TYPE)
  if (_LIBTYPE STREQUAL "SHARED_LIBRARY")
    # Copy the shared library on Windows systems to this directory
    # (examples) so that the tests can be run.
    add_custom_command (TARGET examples POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E
      copy $<TARGET_FILE:${PROJECT_LIBRARIES}> ${CMAKE_CFG_INTDIR}
      COMMENT "Copying shared library to examples directory")
  endif ()
endif ()

if (MSVC)
  # Only install RandomExample and only on Windows systems
  install (TARGETS RandomExample DESTINATION bin)
endif ()

# Put all the tools into a folder in the IDE
set_property (TARGET examples ${EXAMPLES} PROPERTY FOLDER examples)

# Turn on testing
enable_testing ()

# Here are the tests.
add_test (NAME RandomExample0 COMMAND RandomExample)
add_test (NAME RandomSave0 COMMAND RandomSave)
add_test (NAME RandomCoverage0 COMMAND RandomCoverage)
