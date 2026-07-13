cmake_minimum_required(VERSION 3.20)
project(emel_llama_gguf_reference C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_CPU OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)

if(NOT LLAMA_CPP_SOURCE_DIR)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/reference_ref.txt" LLAMA_CPP_REF)
  string(STRIP "${LLAMA_CPP_REF}" LLAMA_CPP_REF)
  include(FetchContent)
  FetchContent_Declare(
    llama_cpp
    URL "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_REF}.tar.gz"
    URL_HASH "SHA256=50fc7ba104cf688d6a2e66f59c252de3e9e084f6d269954ade467aedf951e358"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR ggml
  )
  FetchContent_MakeAvailable(llama_cpp)
  set(LLAMA_CPP_SOURCE_DIR "${llama_cpp_SOURCE_DIR}")
else()
  execute_process(
    COMMAND git -C "${LLAMA_CPP_SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE LLAMA_CPP_ACTUAL_REF
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/reference_ref.txt" LLAMA_CPP_EXPECTED_REF)
  string(STRIP "${LLAMA_CPP_EXPECTED_REF}" LLAMA_CPP_EXPECTED_REF)
  if(NOT LLAMA_CPP_ACTUAL_REF STREQUAL LLAMA_CPP_EXPECTED_REF)
    message(FATAL_ERROR
      "llama.cpp revision ${LLAMA_CPP_ACTUAL_REF} does not match pinned ${LLAMA_CPP_EXPECTED_REF}")
  endif()
  set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory("${LLAMA_CPP_SOURCE_DIR}/ggml" llama-ggml)
endif()

add_executable(llama-gguf-reference main.cpp)
target_link_libraries(llama-gguf-reference PRIVATE ggml-base)
target_compile_options(llama-gguf-reference PRIVATE
  $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-Wall;-Wextra;-Wpedantic;-Werror>
)
