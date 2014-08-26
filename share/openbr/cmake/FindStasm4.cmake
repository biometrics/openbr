# ================================================================
# The Stasm CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(Stasm4 REQUIRED)
#   target_link_libraries(MY_TARGET ${Stasm4_LIBS})
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

find_path(Stasm4_SOURCE_DIR stasm/include/stasm_lib.h ${CMAKE_SOURCE_DIR}/3rdparty/*)

set(stasm_prefix ${CMAKE_BINARY_DIR}/stasm)

# Have to use a name other than stasm so it doesn't conflict with the project itself
ExternalProject_Add(external_stasm
  URL ${Stasm4_SOURCE_DIR}
  PREFIX ${stasm_prefix}
  CMAKE_ARGS "-DCMAKE_PREFIX_PATH=~/Qt/5.3/clang_64/"
  INSTALL_COMMAND ""
  BINARY_DIR "${stasm_prefix}/lib"
  )

ExternalProject_Add_Step(
    external_stasm
    copy_stasm_headers
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${Stasm4_SOURCE_DIR}/stasm/include" ${stasm_prefix}/include/stasm
    COMMENT "Copying stasm headers..."
    DEPENDERS configure
    )

ExternalProject_Add_Step(
    external_stasm
    copy_stasm_data
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${Stasm4_SOURCE_DIR}/data" ${stasm_prefix}/data
    COMMENT "Copying stasm headers..."
    DEPENDERS configure
    )

# We have to fake a library being created to force external_stasm to be built
add_library(fake_stasm UNKNOWN IMPORTED)
add_dependencies(fake_stasm external_stasm)
ExternalProject_Get_Property(external_stasm install_dir)

set(Stasm4_DIR ${install_dir})
include_directories(${install_dir}/include/stasm)
file(GLOB Stasm4_LIBS ${install_dir}/lib/stasm/*.dylib)
