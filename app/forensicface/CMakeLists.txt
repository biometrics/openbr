include_directories(.)
aux_source_directory(. SRC)

# Run QT
include_directories(${CMAKE_CURRENT_BINARY_DIR})
qt4_wrap_cpp(MOC mainwindow.h)
qt4_wrap_ui(UIC mainwindow.ui)

# Build executable
add_executable(forensicface WIN32 ${SRC} ${MOC} ${UIC} ${BR_RESOURCES})
target_link_libraries(forensicface openbr openbr-gui ${QT_LIBRARIES})

# Install
install(TARGETS forensicface RUNTIME DESTINATION bin
                             BUNDLE DESTINATION .)

# Package
set(CPACK_PACKAGE_EXECUTABLES "forensicface" "Forensic Face" ${CPACK_PACKAGE_EXECUTABLES} PARENT_SCOPE)
set(CPACK_CREATE_DESKTOP_LINKS "forensicface" ${CPACK_CREATE_DESKTOP_LINKS} PARENT_SCOPE)
set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\forensicface.exe" ${CPACK_NSIS_INSTALLED_ICON_NAME} PARENT_SCOPE)
