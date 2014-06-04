add_executable(br-enroll br-enroll.cpp ${BR_RESOURCES})
target_link_libraries(br-enroll openbr ${BR_THIRDPARTY_LIBS})
qt5_use_modules(br-enroll ${QT_DEPENDENCIES})
install(TARGETS br-enroll RUNTIME DESTINATION bin)
