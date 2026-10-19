# The on-device separation engine, shared by the app and the bench app.
#
#   sd_engine_library(<target> <source dir> [PROFILE])
#
# builds a static library from <source dir>/demucs/*.cpp and
# <source dir>/engine.cpp. The bench app uses it for this directory (the
# app's engine) and for kept engines in /engines.
cmake_minimum_required(VERSION 3.22.1)
project(stemdeck_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SD_ENGINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen" CACHE PATH "Eigen source tree")
if(NOT EXISTS "${EIGEN_DIR}/Eigen/Dense")
  message(FATAL_ERROR "Eigen not found at ${EIGEN_DIR}. Run: git submodule update --init")
endif()

function(sd_engine_library name src_dir)
  cmake_parse_arguments(ARG "PROFILE" "" "" ${ARGN})
  file(GLOB srcs "${src_dir}/demucs/*.cpp")
  add_library(${name} STATIC ${srcs} "${src_dir}/engine.cpp" "${SD_ENGINE_DIR}/sdprof.cpp")
  target_include_directories(${name} PUBLIC "${EIGEN_DIR}" "${src_dir}/demucs" "${src_dir}")
  target_compile_options(${name} PRIVATE
    -O3 -fno-unsafe-math-optimizations -freciprocal-math -fno-signed-zeros
    -fexceptions -frtti -fvisibility=hidden -fvisibility-inlines-hidden
    -Wno-unused-parameter -Wno-sign-compare)
  target_compile_definitions(${name} PUBLIC NDEBUG EIGEN_NO_DEBUG)
  if(ARG_PROFILE)
    target_compile_definitions(${name} PUBLIC SD_PROFILE)
    target_include_directories(${name} PRIVATE "${SD_ENGINE_DIR}/demucs")
  endif()
endfunction()

sd_engine_library(sdengine_core "${CMAKE_CURRENT_SOURCE_DIR}")

# Desktop tools, used by CI.
if(NOT ANDROID AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(Threads REQUIRED)
  add_executable(sdengine_cli test/engine_cli.cpp)
  target_link_libraries(sdengine_cli sdengine_core Threads::Threads)

  sd_engine_library(sdengine_core_prof "${CMAKE_CURRENT_SOURCE_DIR}" PROFILE)
  add_executable(sdbench_cli test/bench_cli.cpp bench/bench_core.cpp)
  target_include_directories(sdbench_cli PRIVATE bench)
  target_link_libraries(sdbench_cli sdengine_core_prof Threads::Threads)

  # The first released engine (builds 4-10), to check the app's engine against.
  set(SD_BASELINE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../engines/b4")
  if(EXISTS "${SD_BASELINE_DIR}/engine.cpp")
    sd_engine_library(sdengine_baseline "${SD_BASELINE_DIR}")
    add_executable(sdbench_cli_baseline test/bench_cli.cpp bench/bench_core.cpp)
    target_include_directories(sdbench_cli_baseline PRIVATE bench)
    target_link_libraries(sdbench_cli_baseline sdengine_baseline Threads::Threads)
  endif()
endif()
