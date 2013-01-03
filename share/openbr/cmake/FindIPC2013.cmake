# ================================================================
# The Intel Perceptual Computing SDK 2013 CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(IPC2013 REQUIRED)
#   target_link_libraries(MY_TARGET ${IPC2013_LIBS})
# ================================================================

find_path(IPC2013_DIR include/pxcimage.h "C:/Program Files/Intel/PCSDK")
include_directories(${IPC2013_DIR}/include)
link_directories(${IPC2013_DIR}/lib/x64)
set(IPC2013_LIBS libpxc)
if(MSVC)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()
