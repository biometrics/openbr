find_path(LOKI_DIR loki.spec ${CMAKE_SOURCE_DIR}/3rdparty/*)
mark_as_advanced(LOKI_DIR)
include_directories(${LOKI_DIR}/include)
