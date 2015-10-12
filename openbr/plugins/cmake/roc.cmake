set(BR_WITH_ROC OFF CACHE BOOL "Build with the ROC SDK")

if(${BR_WITH_ROC})
  find_package(ROC REQUIRED)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${ROC_LIBS})

  if(WIN32)
    install(FILES ${ROC_DIR}/lib/libroc.so DESTINATION bin)
  else()
    install(FILES ${ROC_DIR}/lib/libroc.so DESTINATION lib)
  endif()

  install(DIRECTORY ${ROC_DIR}/models/ DESTINATION share/openbr/models/pp5)
else()
  set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/classification/roc.cpp)
endif()
