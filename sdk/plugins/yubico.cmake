set(BR_WITH_YUBICO OFF CACHE BOOL "Build YubiKey authentication")

if(${BR_WITH_YUBICO})
  find_package(YubiKey REQUIRED) # For decrypting YubiKeys
  find_package(YKPers REQUIRED)  # For reading YubiKeys

  install(FILES ${YUBIKEY_LICENSE} DESTINATION doc RENAME YubiKey.license)
  install(FILES ${YKPERS_LICENSE} DESTINATION doc RENAME YKPers.license)
  install(FILES ${YKPERS_RULES} DESTINATION share)

  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${YUBIKEY_SRC} ${YKPERS_SRC} plugins/yubico.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${YKPERS_LIBS})
endif()
