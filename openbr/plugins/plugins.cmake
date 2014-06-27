# Optional Appendable CMake Variables:
# BR_THIRDPARTY_PLUGINS     - Additional plugins
# BR_THIRDPARTY_PLUGINS_DIR - Additional folder(s) of plugins
# BR_THIRDPARTY_SRC         - Additional source code needed by a plugin
# BR_THIRDPARTY_LIBS        - Additional libaries needed by a plugin

# Also look for CMake modules in the thirdparty plugins folder(s)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${BR_THIRDPARTY_PLUGINS_DIR})

foreach(DIR ${BR_THIRDPARTY_PLUGINS_DIR})
  file(GLOB PLUGINS ${DIR}/*.cpp ${DIR}/*.h) # Add the headers too so MOC runs on them
  set(BR_THIRDPARTY_PLUGINS ${BR_THIRDPARTY_PLUGINS} ${PLUGINS})
endforeach()

file(GLOB PLUGINS plugins/*.cpp plugins/*.h)
foreach(PLUGIN ${PLUGINS} ${BR_THIRDPARTY_PLUGINS})
  get_filename_component(PLUGIN_BASENAME ${PLUGIN} NAME_WE)
  get_filename_component(PLUGIN_PATH ${PLUGIN} PATH)
  set(PLUGIN_CMAKE "${PLUGIN_PATH}/${PLUGIN_BASENAME}.cmake")
  if(EXISTS ${PLUGIN_CMAKE})
    include(${PLUGIN_CMAKE})
  else()
    set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${PLUGIN})
  endif()
endforeach()
