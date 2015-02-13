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

set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} core)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} io)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} gui)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} imgproc)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} video)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} cluster)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} distance)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} format)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} representation)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} detection)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} classification)
set(BR_PLUGINS_DIR ${BR_PLUGINS_DIR} metadata)


foreach(DIR ${BR_PLUGINS_DIR})
  file(GLOB PLUGINS plugins/${DIR}/*.cpp plugins/${DIR}/*.h)
  set(BR_PLUGINS ${BR_PLUGINS} ${PLUGINS})
endforeach()

file(GLOB PLUGINS plugins/*.cpp plugins/*.h)
set(BR_PLUGINS ${BR_PLUGINS} ${PLUGINS})
foreach(PLUGIN ${BR_PLUGINS} ${BR_THIRDPARTY_PLUGINS})
  get_filename_component(PLUGIN_BASENAME ${PLUGIN} NAME_WE)
  get_filename_component(PLUGIN_PATH ${PLUGIN} PATH)
  set(PLUGIN_CMAKE "${PLUGIN_PATH}/${PLUGIN_BASENAME}.cmake")
  if(EXISTS ${PLUGIN_CMAKE})
    include(${PLUGIN_CMAKE})
  else()
    set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${PLUGIN})
  endif()
endforeach()
