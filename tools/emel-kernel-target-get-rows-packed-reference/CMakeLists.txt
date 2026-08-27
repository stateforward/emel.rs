cmake_minimum_required(VERSION 3.20)

project(emel-kernel-target-get-rows-packed-reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED EMEL_CPP_SOURCE_DIR)
  message(FATAL_ERROR "EMEL_CPP_SOURCE_DIR must name the pinned emel.cpp checkout")
endif()

add_executable(emel-kernel-target-get-rows-packed-reference main.cpp)
target_include_directories(
  emel-kernel-target-get-rows-packed-reference PRIVATE
  "${EMEL_CPP_SOURCE_DIR}/include"
  "${EMEL_CPP_SOURCE_DIR}/src"
  "${FETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML}/include")
