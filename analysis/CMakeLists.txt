# Analysis stages that run after separation: beat grid and key (and, in later
# releases, sections and the lead/backing vocal split). Ports of desktop
# StemDeck's Python, checked against it by scripts/analysis_parity.py.
cmake_minimum_required(VERSION 3.22.1)
project(stemdeck_analysis CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SDA_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../engine/third_party/eigen" CACHE PATH "Eigen source tree")

file(GLOB sda_srcs CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
add_library(sdanalysis STATIC ${sda_srcs})
target_include_directories(sdanalysis PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
  "${SDA_EIGEN_DIR}")
target_compile_options(sdanalysis PRIVATE -O3 -fexceptions -frtti -Wno-sign-compare)
target_link_libraries(sdanalysis PUBLIC ${CMAKE_DL_LIBS})
if(ANDROID)
  target_link_libraries(sdanalysis PUBLIC dl)
endif()

if(NOT ANDROID AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(Threads REQUIRED)
  add_executable(sdanalysis_cli test/cli.cpp)
  target_link_libraries(sdanalysis_cli sdanalysis Threads::Threads)
endif()
