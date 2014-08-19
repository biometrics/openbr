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

find_path(Stasm_DIR stasm/stasm_lib.h ${CMAKE_SOURCE_DIR}/3rdparty/*)

set(CMAKE_ARGS "${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX:PATH=<${Stasm_DIR}>")

ExternalProject_Add(stasm
  URL ${Stasm_DIR}
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  )

ExternalProject_Get_Property(stasm install_dir)

include_directories(${install_dir}/include/stasm)

file(GLOB Stasm4_LIBS ${install_dir}/lib/*)
