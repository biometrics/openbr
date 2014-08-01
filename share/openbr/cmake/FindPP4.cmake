# ================================================================
# The PP4 CMake configuration file
#
# Usage from an external project:
#   In your CMakeLists.txt, add these lines:
#
#   find_package(PP4 REQUIRED)
#   target_link_libraries(MY_TARGET ${PP4_LIBS})
# ================================================================

find_path(PP4_DIR include/pittpatt_nc_sdk.h ${CMAKE_SOURCE_DIR}/3rdparty/*)
include_directories(${PP4_DIR}/include)
link_directories(${PP4_DIR}/lib)
set(PP4_LIBS pittpatt_nc_sdk
             pittpatt_raw_image
             pittpatt_raw_image_io
             pittpatt_recognition_core
             pittpatt_video_io)
