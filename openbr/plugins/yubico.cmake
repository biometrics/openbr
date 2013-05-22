set(BR_WITH_YUBICO OFF CACHE BOOL "Build YubiKey authentication")

if(${BR_WITH_YUBICO})
  find_package(YubiKey REQUIRED) # For decrypting YubiKeys
  find_package(YKPers REQUIRED)  # For reading YubiKeys

  install(FILES ${YUBIKEY_LICENSE} RENAME YubiKey DESTINATION share/openbr/licenses)
  install(FILES ${YKPERS_LICENSE} RENAME YKPers DESTINATION share/openbr/licenses)
  install(FILES ${YKPERS_RULES} DESTINATION share/openbr)

  set(BR_THIRDPARTY_SRC ${BR_THIRDPARTY_SRC} ${YUBIKEY_SRC} ${YKPERS_SRC} plugins/yubico.cpp)
  set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${YKPERS_LIBS})
endif()
