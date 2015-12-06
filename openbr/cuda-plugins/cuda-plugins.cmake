# Optional Appendable CMake Variables:
# BR_THIRDPARTY_CUDA_PLUGINS     - Additional plugins
# BR_THIRDPARTY_CUDA_PLUGINS_DIR - Additional folder(s) of plugins
# BR_EXCLUDED_CUDA_PLUGINS       - Plugins that should not be built
# BR_THIRDPARTY_SRC         - Additional source code needed by a plugin
# BR_THIRDPARTY_LIBS        - Additional libaries needed by a plugin

# Also look for CMake modules in the thirdparty plugins folder(s)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${BR_THIRDPARTY_CUDA_PLUGINS_DIR})

# Gather all of the plugin subdirectories
file(GLOB SUBFILES cuda-plugins/*)
foreach(FILE ${SUBFILES})
  if(IS_DIRECTORY ${FILE})
    set(BR_CUDA_PLUGINS_DIR ${BR_CUDA_PLUGINS_DIR} ${FILE})
  endif()
endforeach()
set(BR_CUDA_PLUGINS_DIR ${BR_CUDA_PLUGINS_DIR} cuda-plugins/) # Remove this when finished with reorg

# Exclude pertinent plugins based on .cmake files
mark_as_advanced(BR_EXCLUDED_CUDA_PLUGINS)
foreach(DIR cuda-plugins/cmake ${BR_THIRDPARTY_CUDA_PLUGINS_DIR})
  file(GLOB CMAKE_FILES ${DIR}/*.cmake)
  foreach(CMAKE_FILE ${CMAKE_FILES})
    if (NOT ${CMAKE_FILE} MATCHES "Find.*cmake")
      include(${CMAKE_FILE})
    endif()
  endforeach()
endforeach()

# Collect all source files except for excluded plugins
foreach(DIR ${BR_CUDA_PLUGINS_DIR} ${BR_THIRDPARTY_CUDA_PLUGINS_DIR})
  get_filename_component(DIR_NAME ${DIR} NAME)
  file(GLOB CUDA_PLUGINS ${DIR}/*.cu ${DIR}/*.cpp ${DIR}/*.h)
  foreach(CUDA_PLUGIN ${CUDA_PLUGINS})
    get_filename_component(CUDA_PLUGIN_NAME ${CUDA_PLUGIN} NAME)
    set(EXCLUDE FALSE)
    foreach(EXCLUDED_CUDA_PLUGIN ${BR_EXCLUDED_CUDA_PLUGINS})
      get_filename_component(EXCLUDED_CUDA_PLUGIN_NAME ${EXCLUDED_CUDA_PLUGIN} NAME)
      if (${CUDA_PLUGIN_NAME} STREQUAL ${EXCLUDED_CUDA_PLUGIN_NAME})
        set(EXCLUDE TRUE)
      endif()
    endforeach()
    if(NOT ${EXCLUDE})
      set(BR_THIRDPARTY_CUDA_PLUGINS ${BR_THIRDPARTY_CUDA_PLUGINS} ${CUDA_PLUGIN})
    endif()
  endforeach()
endforeach()

set(BR_THIRDPARTY_CUDA_SRC ${BR_THIRDPARTY_CUDA_SRC} ${BR_THIRDPARTY_CUDA_PLUGINS})
