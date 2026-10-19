cmake_minimum_required(VERSION 3.22.1)
project(stemdeck_bench CXX)

# Defines sd_engine_library() and the engine being optimized.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../engine" engine EXCLUDE_FROM_ALL)

sd_engine_library(sdengine_opt_prof "${SD_ENGINE_DIR}" PROFILE)
sd_engine_library(sdengine_base "${CMAKE_CURRENT_SOURCE_DIR}/baseline")

find_library(log-lib log)

function(sd_bench_lib name engine_lib kotlin_class)
  add_library(${name} SHARED bench_jni.cpp "${SD_ENGINE_DIR}/bench/bench_core.cpp")
  target_include_directories(${name} PRIVATE "${SD_ENGINE_DIR}/bench")
  target_compile_definitions(${name} PRIVATE SDB_CLASS="${kotlin_class}")
  target_compile_options(${name} PRIVATE -O3 -fexceptions -frtti -fvisibility=hidden)
  target_link_libraries(${name} ${engine_lib} ${log-lib})
  target_link_options(${name} PRIVATE
    "-Wl,-z,max-page-size=16384" "-Wl,--exclude-libs,ALL" "-Wl,-Bsymbolic")
endfunction()

sd_bench_lib(sdbench_opt sdengine_opt_prof "org/stemdeckport/bench/OptNative")
sd_bench_lib(sdbench_base sdengine_base "org/stemdeckport/bench/BaseNative")
