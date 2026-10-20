# QuickJS-NG's command-line runner (qjs), built for the phone as an
# executable named libqjs.so: Android only lets an app run programs from its
# native library directory, and yt-dlp runs qjs as a program to solve
# YouTube's JavaScript challenges (yt-dlp-ejs). MIT, quickjs-ng/quickjs.
cmake_minimum_required(VERSION 3.22)
project(sdqjs C)
include(FetchContent)
FetchContent_Declare(quickjs
    GIT_REPOSITORY https://github.com/quickjs-ng/quickjs.git
    GIT_TAG 6d46d07d04041b40f4f49eaa7fdebe44c314c699  # v0.17.0
)
set(QJS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(quickjs)
# 16 KB pages (Android 15+): every LOAD segment aligned to 16 KB; stripped.
target_link_options(qjs_exe PRIVATE -Wl,-z,max-page-size=16384 -s)
# Written to <build>/out/libqjs.so, which the app's Gradle task copies.
set_target_properties(qjs_exe PROPERTIES OUTPUT_NAME "libqjs.so" SUFFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out")
