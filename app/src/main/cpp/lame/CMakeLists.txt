# LAME (LGPL 2.0+), the MP3 encoder, built as its own shared library
# (libmp3lame.so) so it is linked dynamically.
#
# The source is downloaded once and checked against the published SHA1 of
# lame-3.100.tar.gz. Set LAME_DIR to use an unpacked copy instead.

set(LAME_VERSION 3.100)
set(LAME_SHA1 64c53b1a4d493237cef5e74944912cd9f98e618d)
set(LAME_CACHE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../build/third_party")
set(LAME_DIR "${LAME_CACHE}/lame-${LAME_VERSION}" CACHE PATH "Unpacked LAME source tree")

if(NOT EXISTS "${LAME_DIR}/include/lame.h")
  set(tarball "${LAME_CACHE}/lame-${LAME_VERSION}.tar.gz")
  file(MAKE_DIRECTORY "${LAME_CACHE}")
  set(urls
    "https://downloads.sourceforge.net/project/lame/lame/${LAME_VERSION}/lame-${LAME_VERSION}.tar.gz"
    "https://deb.debian.org/debian/pool/main/l/lame/lame_${LAME_VERSION}.orig.tar.gz")
  set(got FALSE)
  foreach(url IN LISTS urls)
    if(EXISTS "${tarball}")
      file(SHA1 "${tarball}" have)
      if(have STREQUAL LAME_SHA1)
        set(got TRUE)
        break()
      endif()
      file(REMOVE "${tarball}")
    endif()
    message(STATUS "Downloading LAME ${LAME_VERSION} from ${url}")
    file(DOWNLOAD "${url}" "${tarball}" STATUS st TLS_VERIFY ON)
    list(GET st 0 code)
    if(code EQUAL 0)
      file(SHA1 "${tarball}" have)
      if(have STREQUAL LAME_SHA1)
        set(got TRUE)
        break()
      endif()
      message(WARNING "LAME from ${url} has SHA1 ${have}, expected ${LAME_SHA1}")
    else()
      message(WARNING "Downloading ${url} failed: ${st}")
    endif()
  endforeach()
  if(NOT got)
    message(FATAL_ERROR "Could not download lame-${LAME_VERSION}.tar.gz. Download it by hand to ${tarball}.")
  endif()
  file(ARCHIVE_EXTRACT INPUT "${tarball}" DESTINATION "${LAME_CACHE}")
endif()

file(GLOB lame_srcs "${LAME_DIR}/libmp3lame/*.c" "${LAME_DIR}/mpglib/*.c")
add_library(mp3lame SHARED ${lame_srcs})
target_include_directories(mp3lame
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${LAME_DIR}/libmp3lame" "${LAME_DIR}/mpglib"
  PUBLIC "${LAME_DIR}/include")
target_compile_definitions(mp3lame PRIVATE HAVE_CONFIG_H)
target_compile_options(mp3lame PRIVATE -O3 -w -ffp-contract=fast)
if(ANDROID)
  target_link_options(mp3lame PRIVATE "-Wl,-z,max-page-size=16384")
endif()
target_link_libraries(mp3lame m)
