if(NOT ${BR_EMBEDDED})
  aux_source_directory(. SRC)

  # Build executable
  add_executable(br-creator WIN32 ${SRC} ${MOC} ${UIC} ${BR_RESOURCES})
  target_link_libraries(br-creator openbr openbr-gui ${QT_LIBRARIES})

  # Install
  install(TARGETS br-creator RUNTIME DESTINATION bin BUNDLE DESTINATION .)
endif()
