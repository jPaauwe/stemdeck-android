cmake_minimum_required(VERSION 3.22.1)
project(stemdeck_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Eigen is a git submodule pinned to the commit demucs.cpp was written
# against (see third_party/README.md).
set(EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
if(NOT EXISTS "${EIGEN_DIR}/Eigen/Dense")
  message(FATAL_ERROR "Eigen not found at ${EIGEN_DIR}. Run: git submodule update --init")
endif()

file(GLOB DEMUCS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/demucs/*.cpp")

add_library(sdengine_core STATIC ${DEMUCS_SOURCES} engine.cpp)
target_include_directories(sdengine_core PUBLIC
  "${EIGEN_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/demucs"
  "${CMAKE_CURRENT_SOURCE_DIR}")
# Same math flags demucs.cpp's release build uses, minus -march=native.
target_compile_options(sdengine_core PRIVATE
  -O3 -fno-unsafe-math-optimizations -freciprocal-math -fno-signed-zeros
  -fexceptions -frtti -Wno-unused-parameter -Wno-sign-compare)
target_compile_definitions(sdengine_core PUBLIC NDEBUG EIGEN_NO_DEBUG)

if(ANDROID)
  find_library(log-lib log)
  add_library(sdengine SHARED jni.cpp)
  target_link_libraries(sdengine sdengine_core ${log-lib})
  target_compile_options(sdengine PRIVATE -O3 -fexceptions -frtti)
  # 16 KB page alignment, required for apps targeting Android 15+ devices.
  target_link_options(sdengine PRIVATE "-Wl,-z,max-page-size=16384")
else()
  find_package(Threads REQUIRED)
  add_executable(sdengine_cli "${CMAKE_CURRENT_SOURCE_DIR}/../../test/cpp/engine_cli.cpp")
  target_link_libraries(sdengine_cli sdengine_core Threads::Threads)
endif()
