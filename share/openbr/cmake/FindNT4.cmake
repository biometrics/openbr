# ================================================================
# The NT4 CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(NT4 REQUIRED)
#   target_link_libraries(MY_TARGET ${NT4_LIBS})
# ================================================================

set(NT4_DIR "NT4_DIR-NOTFOUND" CACHE PATH "Neurotec Biometric SDK 4.x Directory")

if(UNIX)
  if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(NT4_DIR_LIB ${NT4_DIR}/Lib/Linux_x86_64)
  else("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(NT4_DIR_LIB ${NT4_DIR}/Lib/Linux_x86)
  endif("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
else(UNIX)
  if("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(NT4_DIR_LIB ${NT4_DIR}/Lib/Win64_x64)
  else("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
    set(NT4_DIR_LIB ${NT4_DIR}/Lib/Win32_x86)
  endif("${CMAKE_SIZEOF_VOID_P}" EQUAL "8")
endif(UNIX)

include_directories(${NT4_DIR}/Include)
link_directories(${NT4_DIR_LIB})
set(NT4_LIBS NBiometrics
             NCore
             NLicensing
             NMedia)
