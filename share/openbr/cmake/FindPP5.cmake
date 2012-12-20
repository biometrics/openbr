# ================================================================
# The PP5 CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(PP5 REQUIRED)
#   target_link_libraries(MY_TARGET ${PP5_LIBS})
# ================================================================

find_path(PP5_DIR include/pittpatt_sdk.h ${CMAKE_SOURCE_DIR}/3rdparty/*)
include_directories(${PP5_DIR}/include)
link_directories(${PP5_DIR}/lib)
set(PP5_LIBS pittpatt_raw_image
             pittpatt_raw_image_io
             pittpatt_sdk
             pittpatt_video_io)
