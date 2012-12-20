if(NOT ${BR_EMBEDDED})
  include_directories(.)
  aux_source_directory(. SRC)

  # Build executable
  add_executable(br-gui WIN32 ${SRC} ${MOC} ${UIC} ${WINDOWS_RESOURCES} ${APPLE_RESOURCES})
  set_target_properties(br-gui PROPERTIES AUTOMOC TRUE)
  target_link_libraries(br-gui openbr openbr-gui ${QT_LIBRARIES})

  # Install
  install(TARGETS br-gui RUNTIME DESTINATION bin BUNDLE DESTINATION .)
endif()
