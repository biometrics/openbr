add_executable(br-download br-download.cpp ${BR_RESOURCES})
target_link_libraries(br-download openbr ${BR_THIRDPARTY_LIBS})
qt5_use_modules(br-download ${QT_DEPENDENCIES})
install(TARGETS br-download RUNTIME DESTINATION bin)
