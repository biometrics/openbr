add_executable(br-print br-print.cpp ${BR_RESOURCES})
target_link_libraries(br-print openbr ${BR_THIRDPARTY_LIBS})
qt5_use_modules(br-print ${QT_DEPENDENCIES})
install(TARGETS br-print RUNTIME DESTINATION bin)
