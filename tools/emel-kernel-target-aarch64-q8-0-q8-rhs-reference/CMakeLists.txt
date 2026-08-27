cmake_minimum_required(VERSION 3.20)

project(emel-kernel-target-aarch64-q8-0-q8-rhs-reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED EMEL_CPP_SOURCE_DIR)
  message(FATAL_ERROR "EMEL_CPP_SOURCE_DIR must name the pinned emel.cpp checkout")
endif()

add_subdirectory("${EMEL_CPP_SOURCE_DIR}" emel-cpp EXCLUDE_FROM_ALL)

add_executable(emel-kernel-target-aarch64-q8-0-q8-rhs-reference main.cpp)
target_include_directories(
  emel-kernel-target-aarch64-q8-0-q8-rhs-reference PRIVATE
  "${EMEL_CPP_SOURCE_DIR}/include"
  "${EMEL_CPP_SOURCE_DIR}/src"
  "${FETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML}/include")
target_link_libraries(emel-kernel-target-aarch64-q8-0-q8-rhs-reference PRIVATE emel)
