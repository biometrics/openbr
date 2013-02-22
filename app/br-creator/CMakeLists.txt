if(NOT ${BR_EMBEDDED})
  aux_source_directory(. SRC)

  # Build executable
  add_executable(br-creator WIN32 ${SRC} ${MOC} ${UIC} ${BR_RESOURCES})
  qt5_use_modules(br-creator ${QT_DEPENDENCIES})
  target_link_libraries(br-creator openbr openbr-gui ${BR_THIRDPARTY_LIBS})

  # Install
  install(TARGETS br-creator RUNTIME DESTINATION bin BUNDLE DESTINATION .)
endif()
