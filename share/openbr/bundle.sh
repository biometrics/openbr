#!/bin/bash
BUNDLE_ROOT="`echo "$0" | sed -e 's/\/MacOS\/BPF//'`"
cd ${BUNDLE_ROOT}/Resources/bin
#export "DYLD_LIBRARY_PATH=../lib"
#export "DYLD_FRAMEWORK_PATH=../lib"
./forensicface
