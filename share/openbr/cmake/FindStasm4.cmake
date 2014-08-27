# ================================================================
# The Stasm CMake configuration file
#
# Formatting from: http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(STASM4 REQUIRED)
#   include_directories(STASM4_INCLUDE_DIR)
#   target_link_libraries(MY_TARGET ${STASM4_LIBS})
# ================================================================
include(ExternalProject)

get_cmake_property(CACHE_VARS CACHE_VARIABLES)
foreach(CACHE_VAR ${CACHE_VARS})
  get_property(CACHE_VAR_HELPSTRING CACHE ${CACHE_VAR} PROPERTY HELPSTRING)
  if(CACHE_VAR_HELPSTRING STREQUAL "No help, variable specified on the command line.")
    get_property(CACHE_VAR_TYPE CACHE ${CACHE_VAR} PROPERTY TYPE)
    if(CACHE_VAR_TYPE STREQUAL "UNINITIALIZED")
      set(CACHE_VAR_TYPE)
    else()
      set(CACHE_VAR_TYPE :${CACHE_VAR_TYPE})
    endif()
    set(CMAKE_ARGS "${CMAKE_ARGS} -D${CACHE_VAR}${CACHE_VAR_TYPE}=\"${${CACHE_VAR}}\"")
  endif()
endforeach()

find_path(STASM4_SOURCE_DIR stasm/include/stasm_lib.h ${CMAKE_SOURCE_DIR}/3rdparty/*)

set(STASM4_PREFIX "${CMAKE_BINARY_DIR}/stasm")
set(STASM4_INSTALL_DIR "${CMAKE_BINARY_DIR}/stasm")

# Have to use a name other than stasm so it doesn't conflict with the project itself
ExternalProject_Add(external_stasm
  URL ${STASM4_SOURCE_DIR}
  PREFIX ${STASM4_PREFIX}
  CMAKE_ARGS \"${CMAKE_ARGS}\";
  INSTALL_DIR ${STASM4_INSTALL_DIR}
  BINARY_DIR "${STASM4_PREFIX}/lib"
  )

# We have to fake a library being created to force external_stasm to be built
add_library(fake_stasm UNKNOWN IMPORTED)
add_dependencies(fake_stasm external_stasm)
ExternalProject_Get_Property(external_stasm install_dir)

set(STASM4_FOUND TRUE)
set(STASM4_INCLUDE_DIR ${install_dir}/include)
set(STASM4_DATA_DIR ${install_dir}/src/external_stasm/data)

# We have to explicitly set these, because they aren't created by the time the calling project gets to this point
set(STASM4_LIBS ${install_dir}/lib/stasm/libstasm.dylib ${install_dir}/lib/stasm/libstasm.dylib)
