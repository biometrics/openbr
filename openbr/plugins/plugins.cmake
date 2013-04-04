# Add source to BR_THIRDPARTY_SRC
# Add libs to BR_THIRDPARTY_LIBS

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

