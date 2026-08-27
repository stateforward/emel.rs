cmake_minimum_required(VERSION 3.20)

project(emel-kernel-target-get-rows-reference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED EMEL_CPP_SOURCE_DIR)
  message(FATAL_ERROR "EMEL_CPP_SOURCE_DIR must name the pinned emel.cpp checkout")
endif()

set(EMEL_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(EMEL_ENABLE_FUZZ OFF CACHE BOOL "" FORCE)
option(EMEL_REFERENCE_STANDALONE
       "Compile the header-only reference observer without the full emel target"
       OFF)

add_executable(emel-kernel-target-get-rows-reference main.cpp)
if(EMEL_REFERENCE_STANDALONE)
  target_include_directories(
    emel-kernel-target-get-rows-reference PRIVATE
    "${EMEL_CPP_SOURCE_DIR}/include"
    "${EMEL_CPP_SOURCE_DIR}/src"
    "${FETCHCONTENT_SOURCE_DIR_STATEFORWARD_SML}/include")
else()
  add_subdirectory("${EMEL_CPP_SOURCE_DIR}" emel-cpp EXCLUDE_FROM_ALL)
  target_link_libraries(emel-kernel-target-get-rows-reference PRIVATE emel)
endif()
