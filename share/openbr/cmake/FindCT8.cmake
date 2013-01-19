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


if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
  set(ARCH_STRING x86_64)
else("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
  set(ARCH_STRING x86_32)
endif("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")

if(DEFINED MSVC80)
  set(COMP_DIR_EXT "msc_8.0-ipp_crtdll")
elseif(DEFINED MSVC90)
  set(COMP_DIR_EXT "msc_9.0-ipp_crtdll")
elseif(DEFINED MSVC10)
  set(COMP_DIR_EXT "msc_10.0-ipp_crtdll")
elseif(CMAKE_HOST_APPLE)
  set(COMP_DIR_EXT "gcc-4.2-ipp")
else()
  set(COMP_DIR_EXT "gcc-4.3-ipp")
endif()

set(CT8_DIR_LIB ${CT8_DIR}/lib/${ARCH_STRING}/${COMP_DIR_EXT} )
set(CT8_LIBRARY_RELEASE libfrsdk-8.6.0)
set(CT8_LIBRARY_DEBUG libfrsdk-8.6.0d)

include_directories(${CT8_DIR}/include)
link_directories(${CT8_DIR_LIB} ${CT8_DIR_LIB}_g)
