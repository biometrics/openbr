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

	if(CACHE_VAR STREQUAL "CMAKE_PREFIX_PATH")
		set(CMAKE_PREFIX_PATH "\"${${CACHE_VAR}}\"")
	endif()
  endif()
endforeach()

find_path(STASM4_SOURCE_DIR stasm/include/stasm_lib.h ${CMAKE_SOURCE_DIR}/3rdparty/*)

set(STASM4_PREFIX "${CMAKE_BINARY_DIR}/stasm")
set(STASM4_INSTALL_DIR "${CMAKE_BINARY_DIR}/stasm")

# ExternalProject_Add doesn't respect the ; path separator (i.e. ; don't get no respect)
string(REPLACE ";" "::" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")

# Have to use a name other than stasm so it doesn't conflict with the project itself
ExternalProject_Add(external_stasm
  URL ${STASM4_SOURCE_DIR}
  PREFIX ${STASM4_PREFIX}
  LIST_SEPARATOR ::
  CMAKE_ARGS 	-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
                -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                -DCMAKE_BUILD_TYPE=Release
  INSTALL_DIR ${STASM4_INSTALL_DIR}
  BINARY_DIR "${STASM4_PREFIX}/build"
)

# We have to fake a library being created to force external_stasm to be built
add_library(fake_stasm UNKNOWN IMPORTED)
add_dependencies(fake_stasm external_stasm)
ExternalProject_Get_Property(external_stasm install_dir)

set(STASM4_FOUND TRUE)
set(STASM4_DIR ${install_dir})
set(STASM4_DATA_DIR ${install_dir}/src/external_stasm/data)

link_directories(${install_dir}/lib)

set(STASM4_LIBS stasm)
