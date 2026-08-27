cmake_minimum_required(VERSION 3.20)

project(emel-kernel-aarch64-silu-reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED EMEL_CPP_SOURCE_DIR)
  message(FATAL_ERROR "EMEL_CPP_SOURCE_DIR must name the pinned emel.cpp checkout")
endif()

set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(EMEL_ENABLE_FUZZ OFF CACHE BOOL "" FORCE)
add_subdirectory("${EMEL_CPP_SOURCE_DIR}" emel-cpp EXCLUDE_FROM_ALL)

add_executable(emel-kernel-aarch64-silu-reference main.cpp)
target_link_libraries(emel-kernel-aarch64-silu-reference PRIVATE emel)
