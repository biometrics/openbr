add_executable(br-search br-search.cpp ${BR_RESOURCES})
target_link_libraries(br-search openbr ${BR_THIRDPARTY_LIBS})
qt5_use_modules(br-search ${QT_DEPENDENCIES})
install(TARGETS br-search RUNTIME DESTINATION bin)
