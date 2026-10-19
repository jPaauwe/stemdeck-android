/* Hand-written config.h for building LAME with CMake (Android and host).
   Replaces the one LAME's ./configure would generate. */
#ifndef SD_LAME_CONFIG_H
#define SD_LAME_CONFIG_H

#define STDC_HEADERS 1
#define HAVE_STDINT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_LIMITS_H 1
#define HAVE_STRING_H 1
#define HAVE_STRINGS_H 1
#define HAVE_STDLIB_H 1
#define HAVE_ERRNO_H 1
#define HAVE_FCNTL_H 1
#define HAVE_UNISTD_H 1
#define HAVE_MEMORY_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_SYS_STAT_H 1

/* The decoder (mpglib), used to read our own MP3 stems back for mixing. */
#define HAVE_MPGLIB 1

/* Same defaults ./configure picks. */
#define USE_FAST_LOG 1
#define LAME_LIBRARY_BUILD 1

typedef float ieee754_float32_t;
typedef double ieee754_float64_t;

#endif
