cmake_minimum_required(VERSION 3.26)  # Development.SABIModule / USE_SABI
project(speech_to_phrase LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Stable-ABI (abi3) Python module: build once, run on any CPython >= 3.12.
find_package(Python REQUIRED COMPONENTS Interpreter Development.SABIModule)

# OpenFST (system libfst-dev for dev; auditwheel bundles it into manylinux wheels).
find_library(FST_LIBRARY NAMES fst REQUIRED)
find_path(FST_INCLUDE_DIR NAMES fst/fstlib.h REQUIRED)

python_add_library(_fst MODULE USE_SABI 3.12 WITH_SOABI
  speech_to_phrase/_fst/fstmodule.cc)
target_include_directories(_fst PRIVATE ${FST_INCLUDE_DIR})
target_link_libraries(_fst PRIVATE ${FST_LIBRARY} ${CMAKE_DL_LIBS})

install(TARGETS _fst DESTINATION speech_to_phrase)
