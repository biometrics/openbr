# Stasm CMakeLists

aux_source_directory(stasm/src STASM_SRC)
include_directories(stasm/include)

#aux_source_directory(tasm/src TASM_SRC)
#include_directories(tasm/include)

add_library(stasm SHARED ${STASM_SRC} ${TASM_SRC})
qt5_use_modules(stasm ${QT_DEPENDENCIES})

set_target_properties(stasm PROPERTIES
                                   DEFINE_SYMBOL STASM_LIBRARY
                                   VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}
                                   SOVERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}
                                   LINK_INTERFACE_LIBRARIES ""
                                   AUTOMOC TRUE)

target_link_libraries(stasm ${OpenCV_LIBS} ${Qt5Core_QTMAIN_LIBRARIES})

file(GLOB STASM_HEADERS stasm/include/*.h)
install(FILES ${STASM_HEADERS} DESTINATION include/stasm)
file(GLOB CLASSIC_HEADERS stasm/include/classic/*.mh)
install(FILES ${CLASSIC_HEADERS} DESTINATION include/stasm/classic)
file(GLOB HAT_HEADERS stasm/include/hat/*.mh)
install(FILES ${HAT_HEADERS} DESTINATION include/stasm/hat)
#file(GLOB TASM_HEADERS tasm/include/*.h)
#install(FILES ${TASM_HEADERS} DESTINATION include/tasm)

install(TARGETS stasm RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
