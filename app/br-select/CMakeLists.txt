add_executable(br-select br-select.cpp ${BR_RESOURCES})
target_link_libraries(br-select openbr ${BR_THIRDPARTY_LIBS})
qt5_use_modules(br-select ${QT_DEPENDENCIES})
install(TARGETS br-select RUNTIME DESTINATION bin)
