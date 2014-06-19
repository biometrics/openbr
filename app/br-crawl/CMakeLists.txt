add_executable(br-crawl br-crawl.cpp ${BR_RESOURCES})
target_link_libraries(br-crawl openbr ${BR_THIRDPARTY_LIBS})
qt5_use_modules(br-crawl ${QT_DEPENDENCIES})
install(TARGETS br-crawl RUNTIME DESTINATION bin)
