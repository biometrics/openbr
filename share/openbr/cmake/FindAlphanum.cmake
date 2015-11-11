find_path(ALPHANUM_DIR NAMES alphanum.hpp PATHS ${PROJECT_SOURCE_DIR}/3rdparty/* NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
mark_as_advanced(ALPHANUM_DIR)
include_directories(${ALPHANUM_DIR})
