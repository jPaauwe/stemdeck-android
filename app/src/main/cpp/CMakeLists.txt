cmake_minimum_required(VERSION 3.22.1)
project(stemdeck_app CXX)

# The separation engine lives in /engine, shared with the bench app.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../engine" engine EXCLUDE_FROM_ALL)

find_library(log-lib log)
add_library(sdengine SHARED jni.cpp)
target_link_libraries(sdengine sdengine_core ${log-lib})
target_compile_options(sdengine PRIVATE -O3 -fexceptions -frtti)
# 16 KB page alignment, required for apps targeting Android 15+ devices.
target_link_options(sdengine PRIVATE "-Wl,-z,max-page-size=16384")
