# Stasm CMakeLists

include_directories(${CMAKE_CURRENT_BINARY_DIR})

aux_source_directory(. STASM_SRC)
aux_source_directory(MOD_1 MOD1_SRC)

add_library(stasm SHARED ${STASM_SRC} ${MOD1_SRC})
qt5_use_modules(stasm ${QT_DEPENDENCIES})

set_target_properties(stasm PROPERTIES
                                   DEFINE_SYMBOL STASM_LIBRARY
                                   VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}
                                   SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}
                                   LINK_INTERFACE_LIBRARIES ""
                                   AUTOMOC TRUE)

target_link_libraries(stasm ${OpenCV_LIBS} ${Qt5Core_QTMAIN_LIBRARIES})

file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB HEADERS ${HEADERS} "${CMAKE_CURRENT_SOURCE_DIR}/MOD_1/*.h")

install(FILES ${HEADERS} DESTINATION include/stasm)
install(TARGETS stasm RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
