cmake_minimum_required(VERSION 3.22.1)
project(stemdeck_app C CXX)

# The separation engine lives in /engine, shared with the bench app.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../engine" engine EXCLUDE_FROM_ALL)
# LAME, for MP3 stems (libmp3lame.so).
add_subdirectory(lame)

find_library(log-lib log)
add_library(sdengine SHARED jni.cpp mp3_jni.cpp mp3_codec.cpp)
target_link_libraries(sdengine sdengine_core mp3lame ${log-lib})
target_compile_options(sdengine PRIVATE -O3 -fexceptions -frtti)
# 16 KB page alignment, required for apps targeting Android 15+ devices.
target_link_options(sdengine PRIVATE "-Wl,-z,max-page-size=16384")
