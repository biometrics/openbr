# ================================================================
# The CT8 CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(CT8 REQUIRED)
#   target_link_libraries(MY_TARGET ${CT8_LIBS})
# ================================================================

set(CT8_DIR "CT8_DIR-NOTFOUND" CACHE PATH "Cognitec FaceVACS 8.x directory")

if(CMAKE_HOST_APPLE)
  if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(CT8_DIR_LIB ${CT8_DIR}/lib/x86_64/gcc-4.2-ipp)
  else("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(CT8_DIR_LIB ${CT8_DIR}/lib/x86_32/gcc-4.2-ipp)
  endif("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
else()
  if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(CT8_DIR_LIB ${CT8_DIR}/lib/x86_64/gcc-4.3-ipp)
  else("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(CT8_DIR_LIB ${CT8_DIR}/lib/x86_32/gcc-4.3-ipp)
  endif("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
endif()

set(CT8_LIBS frsdk-8.5.0)
include_directories(${CT8_DIR}/include)
link_directories(${CT8_DIR_LIB})
