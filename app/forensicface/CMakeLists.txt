if(NOT ${BR_EMBEDDED})
  aux_source_directory(. SRC)

  # Run QT
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
  qt5_wrap_ui(UIC mainwindow.ui)

  # Build executable
  add_executable(forensicface WIN32 ${SRC} ${UIC} ${BR_RESOURCES})
  qt5_use_modules(forensicface ${QT_DEPENDENCIES})
  target_link_libraries(forensicface openbr openbr-gui)

  # Install
  install(TARGETS forensicface RUNTIME DESTINATION bin
                             BUNDLE DESTINATION .)

  # Package
  set(CPACK_PACKAGE_EXECUTABLES "forensicface" "Forensic Face" ${CPACK_PACKAGE_EXECUTABLES} PARENT_SCOPE)
  set(CPACK_CREATE_DESKTOP_LINKS "forensicface" ${CPACK_CREATE_DESKTOP_LINKS} PARENT_SCOPE)
  set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\forensicface.exe" ${CPACK_NSIS_INSTALLED_ICON_NAME} PARENT_SCOPE)
endif()
