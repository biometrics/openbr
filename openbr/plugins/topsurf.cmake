set(BR_WITH_TOPSURF OFF CACHE BOOL "Build with TOP-SURF")

if(${BR_WITH_TOPSURF})
  find_package(TopSurf REQUIRED)
  set(THIRDPARTY_SRC ${THIRDPARTY_SRC} plugins/topsurf.cpp ${TOPSURF_SRC} ${TOPSURF_FLANN_SRC})
  install(DIRECTORY ${TOPSURF_DIR}/dictionary_10000
                    ${TOPSURF_DIR}/dictionary_20000
                    ${TOPSURF_DIR}/dictionary_40000
          DESTINATION models/topsurf)
endif()
